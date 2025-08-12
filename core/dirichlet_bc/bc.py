"""
This module provides the `RowDirichletBC` class, which is a flexible tool for
applying Dirichlet boundary conditions to the simulation.

The `RowDirichletBC` class allows for the creation of Dirichlet boundary
conditions along various edges or lines of the simulation domain. It can
handle both constant and time-dependent boundary conditions, and it provides
a convenient way to specify the location and extent of the boundary condition.
"""
import numpy as np
from dolfinx import fem
from petsc4py import PETSc


class RowDirichletBC:
    """Dirichlet BC along one edge/line, optionally clipped to a centred segment.

    Locations
    ---------
    left, right, bottom, top, outer, x, y  (see below)

    Parameters
    ----------
    V : fem.FunctionSpace
        Function space on which the BC acts (typically CG space).
    location : str
        "left", "right", "bottom", "top", "outer", "x", or "y".
    coord : float, optional
        Required for inner-line cases (location "x" or "y"). Ignored otherwise.
    center : float, optional
        Coordinate of the center of the BC. If ``location`` is ``"x"`` or ``"y"``
        and ``center`` is ``None``, the BC is centered in the middle of the domain.
    length : float, optional
        When provided, only DOFs within ``±length/2`` of the midpoint are clamped.
    width : float, optional
        Geometric tolerance when comparing coordinates (default ``1e-12``).
    value : float | callable(x, y, t) -> scalar, optional
        Boundary value. If callable, call :meth:`update` each time step.
    """

    def __init__(self, V, location, *, coord=None, length=None, center=None, width=1e-10, value=0.0):
        self.V = V
        self.mesh = V.mesh
        self.width = float(width)
        self.center = center
        self.length = length

        # Domain extents
        verts = self.mesh.geometry.x
        xmin, ymin = verts[:, 0].min(), verts[:, 1].min()
        xmax, ymax = verts[:, 0].max(), verts[:, 1].max()
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        half = None if length is None else 0.5 * length
        
        if (location in ['x', 'y']) and center is None:
            self.center= xmid if location=='x' else ymid

        # Helper: centred-length mask along an axis array
        def centred_mask(axis_vals, center):
            if half is None:
                return np.ones_like(axis_vals, dtype=bool)
            return np.abs(axis_vals - center) <= half + 1e-14

        # Build vectorised predicate ------------------------------------------------
        if location == "left":
            def pred(x):
                return np.logical_and(
                    np.isclose(x[0], xmin, atol=self.width),
                    centred_mask(x[1], ymid))
        elif location == "right":
            def pred(x):
                return np.logical_and(
                    np.isclose(x[0], xmax, atol=self.width),
                    centred_mask(x[1], ymid))
        elif location == "bottom":
            def pred(x):
                return np.logical_and(
                    np.isclose(x[1], ymin, atol=self.width),
                    centred_mask(x[0], xmid))
        elif location == "top":
            def pred(x):
                return np.logical_and(
                    np.isclose(x[1], ymax, atol=self.width),
                    centred_mask(x[0], xmid))
        elif location == "outer":
            def pred(x):
                mask_left = np.logical_and(np.isclose(x[0], xmin, atol=self.width), centred_mask(x[1], ymid))
                mask_right = np.logical_and(np.isclose(x[0], xmax, atol=self.width), centred_mask(x[1], ymid))
                mask_bottom = np.logical_and(np.isclose(x[1], ymin, atol=self.width), centred_mask(x[0], xmid))
                mask_top = np.logical_and(np.isclose(x[1], ymax, atol=self.width), centred_mask(x[0], xmid))
                return mask_left | mask_right | mask_bottom | mask_top
        elif location == "x":
            if coord is None:
                raise ValueError("coord required when location='x'.")
            c = float(coord)
            def pred(x):
                return np.logical_and(
                    np.isclose(x[0], c, atol=self.width),
                    centred_mask(x[1], self.center))
        elif location == "y":
            if coord is None:
                raise ValueError("coord required when location='y'.")
            c = float(coord)
            def pred(x):
                return np.logical_and(
                    np.isclose(x[1], c, atol=self.width),
                    centred_mask(x[0], self.center))
        else:
            raise ValueError("Unknown location keyword.")

        # Locate DOFs ------------------------------------------------------------
        self.row_dofs = fem.locate_dofs_geometrical(self.V, pred)
        if self.row_dofs.size == 0:
            raise RuntimeError("No DOFs found for requested BC location/length.")

        self.dof_coords = self.V.tabulate_dof_coordinates()[self.row_dofs]

        # Storage function and DirichletBC object --------------------------------
        self._g = fem.Function(self.V)
        self._bc = fem.dirichletbc(self._g, self.row_dofs)

        # Wrap constant into callable if needed ----------------------------------
        if callable(value):
            self._value_callable = value
        else:
            self._value_callable = lambda x, y, t, c=value: c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def bc(self):
        """dolfinx DirichletBC ready for assembly."""
        return self._bc

    def update(self, t):
        # drop z, compute vals
        xy   = self.dof_coords[:, :2]
        vals = np.array([ self._value_callable(x, y, t)
                        for x, y in xy ],
                        dtype=PETSc.ScalarType)

        # write into function
        self._g.x.array[self.row_dofs] = vals
        self._g.x.scatter_forward()

    # ------------------------------------------------------------------
    # Convenience constant BC
    # ------------------------------------------------------------------
    @staticmethod
    def constant(V, location, value, *, coord=None, length=None, width=1e-12):
        bc = RowDirichletBC(V, location, coord=coord, length=length, width=width, value=value)
        bc.update(0.0)
        return bc
    
    # -------------------------------------------------------------------
    # Convenience function to examine BC coords
    # -------------------------------------------------------------------

    @staticmethod
    def describe_row_bcs(bc_list, *, label="Row BC"):
        """Print coordinate bounds for every ``RowDirichletBC`` in ``bc_list``.

        Parameters
        ----------
        bc_list : sequence of RowDirichletBC | fem.DirichletBC
            Mix-and-match is fine; non-Row objects are skipped.
        label : str
            Prefix for each line of output (purely cosmetic).
        """
        for k, bc in enumerate(bc_list):
            # Skip non-RowDirichletBC objects
            if not isinstance(bc, RowDirichletBC):
                continue

            xy = bc.dof_coords              # (n, 3) array, already stored
            x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
            y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
            print(f"{label} #{k}: "
                f"x in [{x_min:.3e}, {x_max:.3e}]  "
                f"y in [{y_min:.3e}, {y_max:.3e}]  "
                f"(n = {xy.shape[0]} DOFs)")

