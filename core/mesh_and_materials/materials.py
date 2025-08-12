"""
This module provides the `Material` class, which is a simple data container
for holding information about the different materials in the simulation.

Each `Material` object stores the name, boundaries, mesh size, and physical
properties of a material. This information is used by the `Mesh` class to
create the simulation mesh and by the `OptimizedSimulationEngine` to set up
the material properties in the simulation.
"""
class Material:
    """Representation of a rectangular material region in the mesh.

    Attributes
    ----------
    name : str
        Unique name of the material.
    boundaries : list[float]
        ``[xmin, xmax, ymin, ymax]`` coordinates defining the region.
    mesh_size : float, optional
        Desired target mesh element size within this region.
    properties : dict
        Dictionary mapping property names to values.
    """

    def __init__(self, name, boundaries, properties=None, mesh_size=None, material_tag=None):
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name)}")
        self.name = name
        if not hasattr(boundaries, '__len__') or len(boundaries) != 4:
            raise ValueError("boundaries must be [xmin,xmax,ymin,ymax]")
        xmin, xmax, ymin, ymax = boundaries
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"Invalid boundaries {boundaries}")
        self.boundaries = [float(xmin), float(xmax), float(ymin), float(ymax)]
        if mesh_size is not None and not isinstance(mesh_size, (int, float)):
            raise TypeError(f"mesh_size must be a number, got {type(mesh_size)}")
        self.mesh_size = float(mesh_size) if mesh_size is not None else None
        self.properties = {} if properties is None else dict(properties)

    def contains(self, x, y):
        """Return True if (x,y) lies inside this material region."""
        xmin, xmax, ymin, ymax = self.boundaries
        return xmin <= x <= xmax and ymin <= y <= ymax

    def __repr__(self):
        return f"Material({self.name!r}, bounds={self.boundaries}, size={self.mesh_size})"
