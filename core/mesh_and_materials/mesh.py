"""
This module provides the `Mesh` class, which is a wrapper around the `gmsh`
library for creating and managing the simulation mesh.

The `Mesh` class is responsible for building a 2D mesh of touching rectangles,
where each rectangle represents a different material. It uses `gmsh`'s Python
API to define the geometry, assign physical groups for materials, and generate
the mesh.

The module also provides a `to_dolfinx()` method for converting the `gmsh`
model into a `dolfinx` mesh, which is the format used by the simulation
engine.
"""
import gmsh

# Optional MPI support – falls back to serial if mpi4py is unavailable
try:
    from mpi4py import MPI  # noqa: F401
    COMM = MPI.COMM_WORLD
except (ModuleNotFoundError, ImportError):
    class _SerialComm:
        rank = 0
        size = 1
        def Barrier(self):
            pass
    COMM = _SerialComm()

SCALE = 1e6          # 1 model unit = 1 µm


class Mesh:
    """Container for a gmsh mesh over a rectangular domain with multiple materials.

    Attributes
    ----------
    name : str
        Name of the mesh.
    boundaries : list[float]
        Domain bounds ``[xmin, xmax, ymin, ymax]``.
    materials : list[Material]
        Collection of material regions.
    default_mesh_size : float
        Default mesh size used outside material regions.
    material_tags : dict
        Mapping of material names to physical group tags.
    """

    # Construction ---------------------------------------------------------------------------
    def __init__(self, name, boundaries, materials):
        if not isinstance(name, str): raise TypeError("name must be a string")
        self.name = name
        if len(boundaries) != 4: raise ValueError("boundaries must be 4 floats")
        self.boundaries = [float(b) for b in boundaries]
        self.materials = list(materials)
        self.material_tags = {}
        self.mesh = None

    # Mesh Validation ------------------------------------------------------------------------
    def _check_mesh(self, base_bounds):
        """Validate the mesh configuration.

        Raises if:
            1) ``base_bounds`` duplicates any material,
            2) two materials have identical bounds, or
            3) any material has non-positive width/height.
        """
        seen = {}
        bb = tuple(round(x, 12) for x in base_bounds)  # 1 pm precision
        seen[bb] = "BASE"

        for m in self.materials:
            bbox = tuple(round(x, 12) for x in m.boundaries)
            if bbox in seen:
                who = seen[bbox]
                raise RuntimeError(
                    f"Duplicate rectangle:\n"
                    f"    {m.name} has boundaries {bbox}\n"
                    f"    already used by {who}"
                )
            seen[bbox] = m.name

        for m in self.materials:
            bx, BX, by, BY = m.boundaries
            dx, dy = BX - bx, BY - by
            if dx <= 0 or dy <= 0:
                raise ValueError(
                    f"{m.name}: invalid rectangle "
                    f"(bx,BX,by,BY) = {m.boundaries} → dx={dx}, dy={dy}"
                )
        # Suppress mesh validation output for batch simulations
        # print('no mesh errors found')


    # Mesh Generation ------------------------------------------------------------------------
    def build_mesh(self):
        """Build a 2‑D mesh of touching rectangles—one per material—using the ``geo`` kernel.

        Each material defines its own surface and physical tag. The mesh is refined
        inside each material to ``mesh_size`` and is coarser elsewhere up to the
        maximum of all material sizes.
        """
        # reset gmsh
        if gmsh.isInitialized():
            gmsh.finalize()

        gmsh.initialize()
        gmsh.model.add(self.name)
        self._check_mesh(self.boundaries)

        if COMM.rank == 0:
            # determine default (coarse) mesh size as max of material sizes
            mat_sizes = [mat.mesh_size for mat in self.materials]
            default_size = max(mat_sizes) if mat_sizes else 1.0

            # 1) create geometry primitives for each material rectangle
            for mat in self.materials:
                bx, BX, by, BY = mat.boundaries
                pa = gmsh.model.geo.addPoint(bx,  by,  0.0)
                pb = gmsh.model.geo.addPoint(BX, by,  0.0)
                pc = gmsh.model.geo.addPoint(BX, BY, 0.0)
                pd = gmsh.model.geo.addPoint(bx,  BY, 0.0)
                la = gmsh.model.geo.addLine(pa, pb)
                lb = gmsh.model.geo.addLine(pb, pc)
                lc = gmsh.model.geo.addLine(pc, pd)
                ld = gmsh.model.geo.addLine(pd, pa)
                cl = gmsh.model.geo.addCurveLoop([la, lb, lc, ld])
                surf = gmsh.model.geo.addPlaneSurface([cl])
                mat._tag = surf

            # sync geometry before groups and fields
            gmsh.model.geo.removeAllDuplicates()
            gmsh.model.geo.synchronize()

            # 2) assign physical groups and build mesh-size fields
            box_fields = []
            for mat in self.materials:
                surf = mat._tag
                pg = gmsh.model.addPhysicalGroup(2, [surf])
                gmsh.model.setPhysicalName(2, pg, mat.name)
                mat.tag = pg
                self.material_tags[mat.name] = pg

                bf = gmsh.model.mesh.field.add('Box')
                bx, BX, by, BY = mat.boundaries
                gmsh.model.mesh.field.setNumber(bf, 'XMin', bx)
                gmsh.model.mesh.field.setNumber(bf, 'XMax', BX)
                gmsh.model.mesh.field.setNumber(bf, 'YMin', by)
                gmsh.model.mesh.field.setNumber(bf, 'YMax', BY)
                # refine inside to material size, coarse outside to default_size
                gmsh.model.mesh.field.setNumber(bf, 'VIn', mat.mesh_size)
                gmsh.model.mesh.field.setNumber(bf, 'VOut', default_size)
                box_fields.append(bf)

            # 3) combine fields via Min and set as background
            if box_fields:
                minf = gmsh.model.mesh.field.add('Min')
                gmsh.model.mesh.field.setNumbers(minf, 'FieldsList', box_fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(minf)

        COMM.Barrier()
        gmsh.model.mesh.generate(2)
        self.mesh = gmsh
        COMM.Barrier()





    # FEniCs Interoperability ------------------------------------------------------------
    def to_dolfinx(self, *, comm=COMM, gdim: int = 2, rank: int = 0):
        """Convert the in-memory gmsh model to a DOLFINx mesh.

        Returns
        -------
        dolfinx.mesh.Mesh
            The converted mesh.
        dolfinx.mesh.MeshTags
            Cell markers associated with the mesh.
        dolfinx.mesh.MeshTags
            Facet markers associated with the mesh.
        """
        if self.mesh is None:
            raise RuntimeError("Mesh not built – call build_mesh() first.")

        from dolfinx.io """
This module provides the `Mesh` class, which is a wrapper around the `gmsh`
library for creating and managing the simulation mesh.

The `Mesh` class is responsible for building a 2D mesh of touching rectangles,
where each rectangle represents a different material. It uses `gmsh`'s Python
API to define the geometry, assign physical groups for materials, and generate
the mesh.

The module also provides a `to_dolfinx()` method for converting the `gmsh`
model into a `dolfinx` mesh, which is the format used by the simulation
engine.
"""
import gmshio  # local import to avoid hard dependency if unused
        return gmshio.model_to_mesh(gmsh.model, comm, rank, gdim)
    
    @staticmethod
    def msh_to_dolfinx(filename: str, *, comm=COMM, gdim: int = 2, rank: int = 0):
        """Load a ``.msh`` file and convert it to a DOLFINx mesh.

        This is a convenience wrapper around ``gmsh.open`` and
        :func:`gmshio.model_to_mesh`.
        """
        gmsh.initialize()
        gmsh.open(filename)
        from dolfinx.io """
This module provides the `Mesh` class, which is a wrapper around the `gmsh`
library for creating and managing the simulation mesh.

The `Mesh` class is responsible for building a 2D mesh of touching rectangles,
where each rectangle represents a different material. It uses `gmsh`'s Python
API to define the geometry, assign physical groups for materials, and generate
the mesh.

The module also provides a `to_dolfinx()` method for converting the `gmsh`
model into a `dolfinx` mesh, which is the format used by the simulation
engine.
"""
import gmshio
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim)
        gmsh.finalize()
        return mesh, cell_tags, facet_tags


    # I/O Helpers ------------------------------------------------------------------------
    def write(self, filename: str):
        """Write the current gmsh mesh to disk (e.g., as ``.msh``)."""
        if self.mesh is None:
            raise RuntimeError("Mesh not built – call build_mesh() first.")
        self.mesh.write(filename)


