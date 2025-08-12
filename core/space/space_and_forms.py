import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc


class Space:
    """High-level wrapper for FEniCSx heat-equation workspaces.

    The class owns the function spaces (CG for the solution, DG0 for material
    data), builds variational forms for transient heat conduction and provides
    convenience helpers for:
      - material-property assignment (piecewise constants on the DG0 space)
      - initial-condition construction
      - system assembly (``assemble_matrix`` / ``assemble_vector``)

    Parameters
    ----------
    mesh_and_tags : dolfinx.mesh.Mesh | tuple
        Either a mesh alone or ``(mesh, cell_tags, facet_tags)`` as returned by
        mesh generators.
    V_family : str, optional
        Finite element family for the temperature space (default ``"Lagrange"``).
    V_degree : int, optional
        Polynomial degree for the temperature space (default ``1``).
    Q_family : str, optional
        Finite element family for coefficient fields (default ``"DG"``).
    Q_degree : int, optional
        Polynomial degree for the coefficient space (default ``0``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, mesh_and_tags, V_family="Lagrange", V_degree=1,
                 Q_family="DG", Q_degree=0):
        """Initialize the space by creating function spaces on the provided mesh.

        Parameters
        ----------
        mesh_and_tags : dolfinx.mesh.Mesh | tuple
            Either a mesh alone or ``(mesh, cell_tags, facet_tags)`` as returned
            by mesh generators.
        V_family : str, optional
            Finite element family for the temperature space.
        V_degree : int, optional
            Polynomial degree for the temperature space.
        Q_family : str, optional
            Finite element family for the coefficient space.
        Q_degree : int, optional
            Polynomial degree for the coefficient space.
        """
        # Unpack mesh and optional tags
        if isinstance(mesh_and_tags, tuple) and len(mesh_and_tags) >= 1:
            self.mesh = mesh_and_tags[0]
            self.cell_tags = mesh_and_tags[1] if len(mesh_and_tags) > 1 else None
            self.facet_tags = mesh_and_tags[2] if len(mesh_and_tags) > 2 else None
        else:
            self.mesh = mesh_and_tags
            self.cell_tags = None
            self.facet_tags = None

        # Build Lagrange and DG spaces
        self.V = fem.functionspace(self.mesh, (V_family, V_degree))
        self.Q = fem.functionspace(self.mesh, (Q_family, Q_degree))

        # Forms cache
        self.a_form = None
        self.L_form = None

        self.a_form_steady = None
        self.L_form_steady = None

    # ------------------------------------------------------------------
    # Variational forms
    # ------------------------------------------------------------------
    def build_variational_forms(self, rho_c, kappa, u_n, dt, r0, f=None):
        """Assemble and store the bilinear and linear forms for the heat equation.

        Parameters
        ----------
        rho_c : fem.Function | fem.Constant
            Volumetric heat capacity field.
        kappa : fem.Function | fem.Constant
            Thermal conductivity field.
        u_n : fem.Function
            Solution at the previous time step.
        dt : float
            Time step size.
        f : fem.Function | fem.Constant, optional
            Source term, defaults to zero.

        Returns
        -------
        tuple
            The assembled bilinear and linear forms ``(a_form, L_form)``.
        """
        x = ufl.SpatialCoordinate(self.mesh)
        r = ufl.sqrt((x[1]-r0)**2)

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        if f is None:
            f = fem.Constant(self.mesh, PETSc.ScalarType(0))

        a = (
            rho_c * u * v *  r * ufl.dx
            + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) *  r * ufl.dx
        )
        L = (
            rho_c * u_n * v *  r * ufl.dx
            + dt * f * v * r *  ufl.dx
        )

        self.a_form = fem.form(a)
        self.L_form = fem.form(L)
        return self.a_form, self.L_form
    
    def build_steady_state_variational_forms(self, kappa, f=None):
        """Assemble and store the bilinear and linear forms for the steady-state heat equation.

        Parameters
        ----------
        kappa : fem.Function | fem.Constant
            Thermal conductivity field.
        f : fem.Function | fem.Constant, optional
            Source term, defaults to zero.

        Returns
        -------
        tuple
            The assembled bilinear and linear forms ``(a_form, L_form)``.
        """
        # Define trial and test functions
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Default to zero source if none provided
        if f is None:
            f = fem.Constant(self.mesh, PETSc.ScalarType(0))

        # Steady-state variational forms: -div(kappa * grad(u)) = f
        a = kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx

        # Store forms for later assembly
        self.a_form_steady = fem.form(a)
        self.L_form_steady = fem.form(L)
        return self.a_form_steady, self.L_form_steady

    # ------------------------------------------------------------------
    # Assembly helpers
    # ------------------------------------------------------------------
    def assemble_matrix(self, bcs):
        """Assemble and return the global matrix ``A`` for the bilinear form.

        Parameters
        ----------
        bcs : list[fem.DirichletBC]
            Dirichlet boundary conditions applied to the system.
        """
        A = fem.petsc.assemble_matrix(self.a_form, bcs=bcs)
        A.assemble()
        return A

    def assemble_vector(self, bcs):
        """Assemble and return the global RHS vector ``b`` for the linear form.

        Parameters
        ----------
        bcs : list[fem.DirichletBC]
            Dirichlet boundary conditions applied to the system.
        """
        b = fem.petsc.assemble_vector(self.L_form)
        fem.petsc.apply_lifting(b, [self.a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs)
        return b

    # ------------------------------------------------------------------
    # Material → DG0 helper
    # ------------------------------------------------------------------
    def assign_material_property(self, materials, property_name):
        """Build a DG0 function containing piecewise constants for a material property.

        Parameters
        ----------
        materials : list[Material]
            Materials with ``.tag`` and ``.properties`` mappings.
        property_name : str
            Name of the property to map.

        Returns
        -------
        fem.Function
            Function in ``Q`` with the property assigned per cell.
        """
        if self.cell_tags is None:
            raise RuntimeError("cell_tags not present – cannot map materials.")
        from dolfinx.fem import locate_dofs_topological

        # Map tags to values
        tag_to_val = {}
        for mat in materials:
            if not hasattr(mat, "tag"):
                raise AttributeError("Material object must have a .tag attribute.")
            if property_name not in mat.properties:
                raise KeyError(f"Property '{property_name}' not found in {mat}.")
            tag_to_val[mat.tag] = mat.properties[property_name]

        values = np.zeros(self.Q.dofmap.index_map.size_local, dtype=PETSc.ScalarType)
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, tdim)

        tags = self.cell_tags.values
        cells = self.cell_tags.indices
        for tag, cell in zip(tags, cells):
            val = tag_to_val.get(tag, 0)
            dofs = locate_dofs_topological(self.Q, tdim, [cell])
            values[dofs] = val

        f_prop = fem.Function(self.Q)
        f_prop.x.array[:] = values
        f_prop.x.scatter_forward()
        return f_prop

    # ------------------------------------------------------------------
    # Initial condition helper
    # ------------------------------------------------------------------
    def initial_condition(self, init, *, name="u0"):
        """Create an initial-condition function in ``V`` from various inputs.

        Parameters
        ----------
        init : float | int | callable | array_like
            Constant value, callable ``f(x, y)`` or array of DOF values.
        name : str, optional
            Optional function name (default ``"u0"``).

        Returns
        -------
        fem.Function
            Function populated with the initial data.
        """
        f_ic = fem.Function(self.V)
        f_ic.name = name

        if isinstance(init, (int, float, np.number)):
            f_ic.interpolate(lambda x: np.full(x.shape[1], init, dtype=PETSc.ScalarType))
            return f_ic

        if callable(init):
            try:
                f_ic.interpolate(init)
            except Exception:
                wrapper = self.vectorize_callable(init)
                f_ic.interpolate(wrapper)
            return f_ic

        arr = np.asarray(init, dtype=PETSc.ScalarType)
        if arr.size != f_ic.x.array.size:
            raise ValueError("Array length does not match number of DOFs.")
        f_ic.x.array[:] = arr
        f_ic.x.scatter_forward()
        return f_ic

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def vectorize_callable(func):
        """Wrap a scalar function ``func(x, y)`` so it accepts point arrays.

        Returns
        -------
        callable
            Function matching :meth:`interpolate`'s signature.
        """
        def wrapper(points):
            xx, yy = points[0], points[1]
            return np.array([func(xi, yi) for xi, yi in zip(xx, yy)], dtype=PETSc.ScalarType)
        return wrapper
