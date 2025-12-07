"""
This module provides the `OptimizedSimulationEngine` class, which is the core
of the heat flow simulation.

The engine is designed to be highly configurable via a YAML file, and it
handles all aspects of the simulation, including:

- Setting up the mesh and materials.
- Applying boundary conditions.
- Solving the heat equation using the finite element method.
- Saving the results to disk.

The engine can be run in two modes: a full simulation mode that includes disk
I/O, and a minimal, in-memory mode that is optimized for uncertainty
quantification tasks.
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import contextlib
import traceback

from .mesh_and_materials.mesh import Mesh, COMM
from .mesh_and_materials.materials import Material
from .dirichlet_bc.bc import RowDirichletBC
from dolfinx import fem
import ufl
from petsc4py import PETSc


@contextlib.contextmanager
def suppress_output(enabled: bool):
    """Context manager to suppress stdout/stderr."""
    if not enabled:
        yield
    else:
        try:
            # Save original file descriptors
            old_stdout = os.dup(1)
            old_stderr = os.dup(2)
            
            # Redirect to /dev/null
            with open(os.devnull, 'w') as fnull:
                os.dup2(fnull.fileno(), 1)
                os.dup2(fnull.fileno(), 2)
                try:
                    yield
                finally:
                    # Restore original file descriptors
                    os.dup2(old_stdout, 1)
                    os.dup2(old_stderr, 2)
                    os.close(old_stdout)
                    os.close(old_stderr)
        except (OSError, AttributeError):
            # Fallback for systems where dup2 doesn't work
            with open(os.devnull, 'w') as fnull:
                with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                    yield


class OptimizedSimulationEngine:
    """Optimized simulation engine with YAML configuration support."""
    
    def __init__(self, cfg, mesh_folder, output_folder, config_path=None):
        self.cfg = cfg
        self.mesh_folder = mesh_folder
        self.output_folder = output_folder
        self.config_path = config_path or "configuration"
        
    def run(self, rebuild_mesh=False, visualize_mesh=False, suppress_print=False):
        """
        Run the optimized heat flow simulation.
        
        Parameters:
        -----------
        rebuild_mesh : bool, optional
            Whether to rebuild the mesh and update material tags
        visualize_mesh : bool, optional
            Whether to visualize the mesh (overrides config setting)
        suppress_print : bool, optional
            If True, suppress all print output
        """
        with suppress_output(suppress_print):
            # Get visualization setting from config or parameter
            config_visualize = self.cfg.get('performance', {}).get('visualize_mesh', False)
            should_visualize = visualize_mesh or config_visualize
            
            print("Starting optimized heat flow simulation...")
            print(f"Configuration file: {self.config_path}")
            print(f"Mesh folder: {self.mesh_folder}")
            print(f"Output folder: {self.output_folder}")
            print(f"Rebuild mesh: {rebuild_mesh}")
            print(f"Visualize mesh: {should_visualize}")
            
            # Setup mesh and materials
            domain, cell_tags, materials = self._setup_mesh_and_materials(rebuild_mesh)
            
            # Print material boundaries once
            print("Material boundaries:")
            for name, mat in materials.items():
                bnd = mat.boundaries
                print(f"  {name}: z=[{bnd[0]:.2e}, {bnd[1]:.2e}], r=[{bnd[2]:.2e}, {bnd[3]:.2e}]")
            
            # Visualize mesh if requested
            if should_visualize:
                self._visualize_mesh()
            
            # Setup function spaces and material properties
            V, Q, kappa, rho_cv = self._setup_function_spaces(domain, cell_tags, materials)
            
            # Load heating data
            heating_data = self._load_heating_data()
            
            # Setup boundary conditions
            obj_bcs, bcs, heating_bc_obj = self._setup_boundary_conditions(V, materials, heating_data)
            
            # Initialize solution
            u_n = fem.Function(V)
            ic_temp = float(self.cfg['heating']['ic_temp'])
            u_n.x.array[:] = np.full_like(u_n.x.array, ic_temp)
            u_n.x.scatter_forward()
            u_n.name = 'Temperature (K)'
            
            # Setup variational forms
            lhs_form, rhs_form = self._setup_variational_forms(domain, V, Q, kappa, rho_cv, u_n)
            
            # Setup solver
            solver = self._setup_solver(lhs_form, bcs)
            
            # Setup output
            xdmf_file, watcher_setup = self._setup_output(domain, heating_bc_obj)
            
            # Run time stepping
            timing_results = self._run_time_stepping(
                domain, V, lhs_form, rhs_form, obj_bcs, bcs, solver, u_n, xdmf_file, watcher_setup
            )
            
            # Save watcher data
            self._save_watcher_data(watcher_setup)
            
            # Close XDMF file
            if xdmf_file:
                xdmf_file.close()
            
            # Return results
            return {
                'timing': timing_results,
                'num_steps': int(self.cfg['timing']['num_steps']),
                'watcher_points': len(watcher_setup['names']) if watcher_setup else 0
            }
    
    def _setup_mesh_and_materials(self, rebuild_mesh):
        """Setup mesh and materials with caching."""
        # Check if we can use cached mesh
        if not rebuild_mesh and self._can_use_cached_mesh():
            return self._load_cached_mesh()
        
        # Build new mesh using configuration
        materials = self._create_materials_from_config()
        
        # Calculate mesh boundaries
        mesh_bounds = self._calculate_mesh_boundaries(materials)
        
        # Create mesh
        gmsh_domain = Mesh(
            name='mesh.msh',
            boundaries=mesh_bounds,
            materials=list(materials.values())
        )
        
        # Build and save mesh
        gmsh_domain.build_mesh()
        self._save_mesh_and_config(gmsh_domain, materials)
        
        # Convert to DOLFINx
        from dolfinx.io import gmshio
        import gmsh
        
        gmsh.initialize()
        mesh_file_path = os.path.join(self.mesh_folder, 'mesh.msh')
        gmsh.open(mesh_file_path)
        domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, COMM, 0, 2)
        gmsh.finalize()
        
        return domain, cell_tags, materials
    
    def _create_materials_from_config(self):
        """Create Material objects from configuration - agnostic to material names."""
        materials = {}
        mats_cfg = self.cfg['mats']
        
        # Get material order from config, or use all materials in config
        material_order = self.cfg.get('layout', {}).get('materials', list(mats_cfg.keys()))
        
        # Validate that all materials in order exist in config
        missing_materials = [name for name in material_order if name not in mats_cfg]
        if missing_materials:
            raise ValueError(f"Materials specified in layout not found in config: {missing_materials}")
        
        # Calculate boundaries for stacked layout
        boundaries = self._calculate_material_boundaries(material_order)
        
        for name in material_order:
            mat_cfg = mats_cfg[name]
            materials[name] = Material(
                name=name,
                boundaries=boundaries[name],
                properties={
                    "rho_cv": float(mat_cfg['rho_cv']),
                    "k": float(mat_cfg['k'])
                },
                mesh_size=float(mat_cfg['mesh'])
            )
        
        return materials
    
    def _calculate_material_boundaries(self, material_order):
        """Calculate material boundaries for stacked layout - matching original code."""
        mats_cfg = self.cfg['mats']
        
        # Extract dimensions
        z_dims = {name: float(mats_cfg[name]['z']) for name in material_order}
        r_dims = {name: float(mats_cfg[name]['r']) for name in material_order}
        
        # Find the sample material (center material)
        sample_material = None
        for name in material_order:
            if 'sample' in name.lower():
                sample_material = name
                break
        
        if not sample_material:
            # Fallback: use the middle material
            sample_material = material_order[len(material_order) // 2]
        
        # Calculate mesh boundaries like original code
        z_sample = z_dims[sample_material]
        
        # Calculate materials before and after sample
        pre_sample_materials = []
        post_sample_materials = []
        found_sample = False
        
        for name in material_order:
            if name == sample_material:
                found_sample = True
            elif found_sample:
                post_sample_materials.append(name)
            else:
                pre_sample_materials.append(name)
        
        # Calculate total thickness before and after sample
        pre_sample_z = sum(z_dims[name] for name in pre_sample_materials)
        post_sample_z = sum(z_dims[name] for name in post_sample_materials)
        
        # Calculate mesh boundaries (sample centered at z=0)
        mesh_zmin = -(z_sample/2) - pre_sample_z
        mesh_zmax = (z_sample/2) + post_sample_z
        mesh_rmin = 0.0
        mesh_rmax = max(r_dims.values())
        
        # Calculate material boundaries
        boundaries = {}
        current_z = mesh_zmin
        
        for mat_name in material_order:
            z_size = z_dims[mat_name]
            r_size = r_dims[mat_name]
            
            boundaries[mat_name] = [
                current_z,                    # zmin
                current_z + z_size,           # zmax
                mesh_rmin,                    # rmin
                mesh_rmin + r_size            # rmax
            ]
            current_z += z_size
            
        return boundaries
    
    def _calculate_mesh_boundaries(self, materials):
        """Calculate overall mesh boundaries from materials."""
        if not materials:
            return [0, 1, 0, 1]  # Default bounds
        
        z_mins = [mat.boundaries[0] for mat in materials.values()]
        z_maxs = [mat.boundaries[1] for mat in materials.values()]
        r_mins = [mat.boundaries[2] for mat in materials.values()]
        r_maxs = [mat.boundaries[3] for mat in materials.values()]
        
        return [min(z_mins), max(z_maxs), min(r_mins), max(r_maxs)]
    
    def _can_use_cached_mesh(self):
        """Check if cached mesh can be used."""
        mesh_file = os.path.join(self.mesh_folder, 'mesh.msh')
        config_file = os.path.join(self.mesh_folder, 'mesh_cfg.yaml')
        
        return os.path.exists(mesh_file) and os.path.exists(config_file)
    
    def _load_cached_mesh(self):
        """Load cached mesh and materials."""
        from dolfinx.io import gmshio
        import gmsh
        
        # Load mesh
        gmsh.initialize()
        mesh_file_path = os.path.join(self.mesh_folder, 'mesh.msh')
        gmsh.open(mesh_file_path)
        domain, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, COMM, 0, 2)
        gmsh.finalize()
        
        # Load materials from config
        with open(os.path.join(self.mesh_folder, 'mesh_cfg.yaml'), 'r') as f:
            mesh_cfg = yaml.safe_load(f)
        
        materials = self._create_materials_from_config()
        
        return domain, cell_tags, materials
    
    def _save_mesh_and_config(self, gmsh_domain, materials):
        """Save mesh and configuration."""
        os.makedirs(self.mesh_folder, exist_ok=True)
        
        # Save mesh
        mesh_file_path = os.path.join(self.mesh_folder, 'mesh.msh')
        gmsh_domain.write(mesh_file_path)
        
        # Save config with material tags
        mesh_cfg = self.cfg.copy()
        mat_tag_map = {mat.name: getattr(mat, '_tag', i+1) for i, mat in enumerate(materials.values())}
        mesh_cfg['material_tags'] = mat_tag_map
        
        config_file_path = os.path.join(self.mesh_folder, 'mesh_cfg.yaml')
        with open(config_file_path, 'w') as f:
            yaml.safe_dump(mesh_cfg, f)
    
    def _setup_function_spaces(self, domain, cell_tags, materials):
        """Setup function spaces and material properties."""
        V = fem.functionspace(domain, ("Lagrange", 1))
        Q = fem.functionspace(domain, ("DG", 0))
        
        # Create material tag mapping from materials object
        mat_tag_map = {mat.name: getattr(mat, 'tag', i+1) for i, mat in enumerate(materials.values())}
        
        # Efficient material property assignment using actual mesh tags
        tag_to_k = {mat_tag_map[mat.name]: mat.properties["k"] for mat in materials.values()}
        tag_to_rho_cv = {mat_tag_map[mat.name]: mat.properties["rho_cv"] for mat in materials.values()}
        
        # Vectorized assignment
        cell_tag_array = cell_tags.values
        kappa_per_cell = np.array([tag_to_k[tag] for tag in cell_tag_array])
        rho_cv_per_cell = np.array([tag_to_rho_cv[tag] for tag in cell_tag_array])
        
        kappa = fem.Function(Q)
        rho_cv = fem.Function(Q)
        
        kappa.x.array[:] = kappa_per_cell
        rho_cv.x.array[:] = rho_cv_per_cell
        kappa.x.scatter_forward()
        rho_cv.x.scatter_forward()
        
        return V, Q, kappa, rho_cv
    
    def _load_heating_data(self):
        """Load and preprocess heating data."""
        heating_file = self.cfg['heating']['file']
        df_heat = pd.read_csv(heating_file)
        
        df_heat = (df_heat
                    .sort_values('time')
                    .assign(
                        time=pd.to_numeric(df_heat['time'], errors='coerce'),
                        temp=pd.to_numeric(df_heat['temp'], errors='coerce')
                    )
                    .dropna(subset=['time', 'temp'])
                    .reset_index(drop=True))
        
        # Apply Savitzky-Golay smoothing if configured
        smoothing_cfg = self.cfg.get('heating', {}).get('smoothing', {})
        if smoothing_cfg.get('enabled', False):
            try:
                from scipy.signal import savgol_filter
                
                # Get smoothing parameters
                window_length = smoothing_cfg.get('window_length', 11)
                polyorder = smoothing_cfg.get('polyorder', 3)
                
                # Ensure window_length is odd and not larger than data length
                if window_length % 2 == 0:
                    window_length += 1
                if window_length > len(df_heat):
                    window_length = len(df_heat) if len(df_heat) % 2 == 1 else len(df_heat) - 1
                
                # Apply Savitzky-Golay filter
                temp_smoothed = savgol_filter(df_heat['temp'].values, window_length, polyorder)
                
                # Store both raw and smoothed data
                df_heat['temp_raw'] = df_heat['temp'].copy()
                df_heat['temp'] = temp_smoothed
                
                # Save processed data to output folder for plotting
                processed_data_path = os.path.join(self.output_folder, 'processed_experimental_data.csv')
                df_heat.to_csv(processed_data_path, index=False)
                
                print(f"Applied Savitzky-Golay smoothing: window={window_length}, polyorder={polyorder}")
                print(f"Processed experimental data saved to: {processed_data_path}")
                
            except ImportError:
                print("Warning: scipy.signal not available, skipping Savitzky-Golay smoothing")
            except Exception as e:
                print(f"Warning: Error applying Savitzky-Golay smoothing: {e}")
        else:
            # Smoothing disabled: df_heat['temp'] contains raw data from CSV
            # This raw data will be used directly in the heating boundary condition
            # via linear interpolation (np.interp) in _setup_boundary_conditions()
            print("Smoothing disabled: using raw experimental data for heating boundary condition")
        
        return df_heat
    
    def _setup_boundary_conditions(self, V, materials, heating_data):
        """Setup boundary conditions from configuration - agnostic to material names."""
        obj_bcs = []  # Custom BC objects for updating
        ic_temp = float(self.cfg['heating']['ic_temp'])
        
        # Setup outer boundary conditions
        outer_bcs = self.cfg.get('boundary_conditions', {}).get('outer', {})
        for boundary, config in outer_bcs.items():
            if config.get('type') == 'dirichlet':
                value = config.get('value', ic_temp)
                if value == 'ic_temp':
                    value = ic_temp
                    
                bc = RowDirichletBC(V, boundary, value=float(value))
                obj_bcs.append(bc)
        
        # Setup heating boundary condition
        heating_cfg = self.cfg.get('heating', {})
        bc_cfg = heating_cfg.get('bc', {})
        
        if bc_cfg.get('type') == 'gaussian':
            # Create heating interpolation function
            # NOTE: When smoothing.enabled is False, heating_data['temp'] contains raw data
            #       When smoothing.enabled is True, heating_data['temp'] contains smoothed data
            #       np.interp() performs linear interpolation onto the simulation time grid
            heating_interp = lambda t: np.interp(
                t, 
                heating_data['time'], 
                heating_data['temp'],
                left=heating_data['temp'].iloc[0],
                right=heating_data['temp'].iloc[-1]
            )
            
            fwhm = float(heating_cfg['fwhm'])
            
            # Calculate offset and Gaussian parameters using configurable baseline
            baseline_temp = self._compute_baseline(heating_data['time'].values,
                                                   heating_data['temp'].values)
            offset = baseline_temp - ic_temp
            coeff = -4.0 * np.log(2.0) / fwhm**2
            center = bc_cfg.get('center', 0.0)
            
            def gaussian_heating(x, y, t):
                amp = heating_interp(t) - offset
                return (amp - ic_temp) * np.exp(coeff * (y - center)**2) + ic_temp
            
            # Find the appropriate material for inner boundary - agnostic to material names
            heating_material = bc_cfg.get('material', None)
            
            if heating_material and heating_material in materials:
                # Use specified material
                inner_mat = materials[heating_material]
            else:
                # Fallback: find first material (could be improved with better heuristics)
                inner_mat = list(materials.values())[0]
            
            # Use the z-coordinate of the material boundary (like original code)
            # coord = inner_mat.boundaries[0]  # zmin of material
            
            # Determine which edge to apply the BC on
            bc_location = bc_cfg.get('location', 'zmin').lower()
            if bc_location == 'zmin':
                coord = inner_mat.boundaries[0]  # zmin of material
            elif bc_location == 'zmax':
                coord = inner_mat.boundaries[1]  # zmax of material
            else:
                print(f"Warning: Invalid heating.bc.location '{bc_location}'. Defaulting to 'zmin'.")
                coord = inner_mat.boundaries[0]

            # Calculate length for BC segment - use the sample radius (like original code)
            # Find the sample material to get its radius
            sample_material = None
            for name in self.cfg['mats'].keys():
                if 'sample' in name.lower():
                    sample_material = name
                    break
            
            if sample_material:
                sample_r = float(self.cfg['mats'][sample_material]['r'])
                length = abs(sample_r) * 2
            else:
                # Fallback: use the heating material radius
                heating_r = float(self.cfg['mats'][heating_material]['r'])
                length = abs(heating_r) * 2
            
            # Print boundary condition info
            print(f"Heating BC: material={heating_material}, coord={coord:.2e}, length={length:.2e}, location={bc_location}")
            
            # Create RowDirichletBC with location='x' and coord parameter (like original)
            try:
                bc = RowDirichletBC(
                    V,
                    'x',  # location='x' creates a row along constant x-coordinate
                    coord=coord,  # The z-coordinate where to apply the BC
                    length=length,
                    center=center,
                    width=1e-10,  # Increased tolerance for finding DOFs
                    value=gaussian_heating
                )
                obj_bcs.append(bc)
            except RuntimeError as e:
                if "No DOFs found" in str(e):
                    # Try with even larger tolerance
                    print(f"Warning: No DOFs found at coord={coord:.2e}, trying with larger tolerance...")
                    bc = RowDirichletBC(
                        V,
                        'x',
                        coord=coord,
                        length=length,
                        center=center,
                        width=1e-6,  # Much larger tolerance
                        value=gaussian_heating
                    )
                    obj_bcs.append(bc)
                else:
                    raise
        
        # Convert to DOLFINx BC objects
        bcs = [bc.bc for bc in obj_bcs]
        
        # Return heating BC object separately for watcher point setup
        heating_bc_obj = None
        if bc_cfg.get('type') == 'gaussian':
            # The heating BC object is the last one added (after outer BCs)
            # Find it by checking if it has row_dofs and is a callable BC
            for bc_obj in reversed(obj_bcs):  # Check from end (heating BC is added last)
                if hasattr(bc_obj, 'row_dofs') and callable(bc_obj._value_callable):
                    heating_bc_obj = bc_obj
                    break
        
        return obj_bcs, bcs, heating_bc_obj
    
    def _setup_variational_forms(self, domain, V, Q, kappa, rho_cv, u_n):
        """Setup variational forms for the heat equation."""
        x = ufl.SpatialCoordinate(domain)
        r = x[1]  # y-coord is radial direction
        
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = fem.Constant(domain, PETSc.ScalarType(0))
        
        # Get timing parameters
        t_final = float(self.cfg['timing']['t_final'])
        num_steps = int(self.cfg['timing']['num_steps'])
        dt = t_final / num_steps
        
        lhs = (
            rho_cv * u * v * r * ufl.dx
            + dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * r * ufl.dx
        )
        rhs = (
            rho_cv * u_n * v * r * ufl.dx
            + dt * f * v * r * ufl.dx
        )
        
        lhs_form = fem.form(lhs)
        rhs_form = fem.form(rhs)
        
        return lhs_form, rhs_form
    
    def _setup_solver(self, lhs_form, bcs):
        """Setup PETSc solver based on configuration."""
        solver_cfg = self.cfg.get('solver', {})
        solver_type = solver_cfg.get('type', 'direct')
        
        # Use correct DOLFINx import
        from dolfinx.fem.petsc import assemble_matrix, create_vector
        
        A = assemble_matrix(lhs_form, bcs=bcs)
        A.assemble()
        
        solver = PETSc.KSP().create(A.getComm())
        solver.setOperators(A)
        
        if solver_type == 'direct':
            solver.setType(PETSc.KSP.Type.PREONLY)
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.LU)
            
            factorization = solver_cfg.get('direct', {}).get('factorization', 'mumps')
            pc.setFactorSolverType(factorization)
        
        return solver
    
    def _setup_output(self, domain, heating_bc_obj=None):
        """Setup output files and watcher points.
        
        Parameters:
        -----------
        domain : dolfinx.mesh.Mesh
            The simulation domain
        heating_bc_obj : RowDirichletBC, optional
            The heating boundary condition object (for finding exact boundary DOF)
        """
        output_cfg = self.cfg.get('output', {})
        
        # Setup XDMF output
        xdmf_file = None
        if output_cfg.get('xdmf', {}).get('enabled', True):
            from dolfinx import io
            
            filename = output_cfg['xdmf'].get('filename', 'output.xdmf')
            xdmf_path = os.path.join(self.output_folder, filename)
            
            xdmf_file = io.XDMFFile(domain.comm, xdmf_path, "w")
            xdmf_file.write_mesh(domain)
        
        # Setup watcher points
        watcher_setup = None
        if output_cfg.get('watcher_points', {}).get('enabled', True):
            watcher_setup = self._setup_watcher_points(domain, output_cfg['watcher_points'], heating_bc_obj)
        
        return xdmf_file, watcher_setup
    
    def _setup_watcher_points(self, domain, watcher_cfg, heating_bc_obj=None):
        """Setup watcher points for temperature monitoring - agnostic to material names.
        
        Parameters:
        -----------
        domain : dolfinx.mesh.Mesh
            The simulation domain
        watcher_cfg : dict
            Watcher points configuration
        heating_bc_obj : RowDirichletBC, optional
            The heating boundary condition object (for finding exact boundary DOF)
        """
        points_cfg = watcher_cfg.get('points', {})
        
        # Calculate watcher point coordinates
        watcher_coords = []
        watcher_names = []
        
        for name, config in points_cfg.items():
            material_name = config['material']
            position = config['position']
            
            # Validate material exists
            if material_name not in self.cfg['mats']:
                raise ValueError(f"Watcher point material '{material_name}' not found in configuration")
            
            # Get material boundaries from config
            mat_cfg = self.cfg['mats'][material_name]
            z_size = float(mat_cfg['z'])
            r_size = float(mat_cfg['r'])
            
            # Calculate position based on material layout
            material_order = self.cfg.get('layout', {}).get('materials', list(self.cfg['mats'].keys()))
            boundaries = self._calculate_material_boundaries(material_order)
            
            if material_name not in boundaries:
                raise ValueError(f"Material '{material_name}' not found in layout")
            
            bnd = boundaries[material_name]
            mat_zmin, mat_zmax, mat_rmin, mat_rmax = bnd
            
            # Handle different position specifications
            if position == 'center':
                # Center of material
                z_coord = (mat_zmin + mat_zmax) / 2
                r_coord = (mat_rmin + mat_rmax) / 2
            elif isinstance(position, (list, tuple)) and len(position) == 2:
                # Explicit coordinates [z, r]
                z_coord, r_coord = position
            elif isinstance(position, dict):
                # New percentage-based positioning
                r_value = position.get('r', (mat_rmin + mat_rmax) / 2)  # Default to center r
                percentage = position.get('percentage', 0.5)  # Default to center
                
                # Validate percentage
                if not 0.0 <= percentage <= 1.0:
                    raise ValueError(f"Percentage must be between 0.0 and 1.0, got {percentage}")
                
                # Calculate z-coordinate based on percentage within material
                z_coord = mat_zmin + percentage * (mat_zmax - mat_zmin)
                r_coord = r_value
            elif position == 'heating_location':
                # Special case: place watcher at the heating boundary condition location
                heating_cfg = self.cfg.get('heating', {}).get('bc', {})
                heating_material = heating_cfg.get('material')
                
                if heating_material and heating_material in self.cfg['mats']:
                    # Get the heating material boundaries
                    heating_boundaries = self._calculate_material_boundaries(
                        self.cfg.get('layout', {}).get('materials', list(self.cfg['mats'].keys()))
                    )
                    if heating_material in heating_boundaries:
                        # Use the same logic as boundary condition setup to determine zmin vs zmax
                        bc_location = heating_cfg.get('location', 'zmin').lower()
                        if bc_location == 'zmin':
                            z_coord = heating_boundaries[heating_material][0]  # zmin
                        elif bc_location == 'zmax':
                            z_coord = heating_boundaries[heating_material][1]  # zmax
                        else:
                            print(f"Warning: Invalid heating.bc.location '{bc_location}'. Defaulting to 'zmin'.")
                            z_coord = heating_boundaries[heating_material][0]  # zmin
                        r_coord = 0.0  # Center of radial direction
                    else:
                        raise ValueError(f"Heating material '{heating_material}' not found in layout")
                else:
                    raise ValueError("Heating material not specified in configuration")
            else:
                raise ValueError(f"Invalid position specification: {position}. Use 'center', [z, r], 'heating_location', or {{'r': r_value, 'percentage': p}}")
            
            watcher_coords.append((z_coord, r_coord))
            watcher_names.append(name)
        
        # Find watcher nodes - use exact boundary DOF for heating_location if available
        mesh_coords = domain.geometry.x[:, :2]
        tree = cKDTree(mesh_coords)
        
        watcher_nodes = []
        watcher_use_bc_value = []  # Track which watchers should use BC value directly
        watcher_bc_funcs = []  # Store BC functions for direct value recording
        
        for i, (name, coords) in enumerate(zip(watcher_names, watcher_coords)):
            z_coord, r_coord = coords
            
            # Check if this is a heating_location watcher and we have the BC object
            if (name in points_cfg and 
                points_cfg[name].get('position') == 'heating_location' and 
                heating_bc_obj is not None):
                
                # Find the exact boundary DOF at r=0 (or closest to r=0)
                bc_dofs = heating_bc_obj.row_dofs
                bc_dof_coords = heating_bc_obj.dof_coords[:, :2]  # (z, r) coordinates
                
                # Find DOF closest to r=0 (within tolerance)
                r_tolerance = 1e-8
                r_distances = np.abs(bc_dof_coords[:, 1] - r_coord)  # r-coordinate differences
                valid_dofs = np.where(r_distances <= r_tolerance)[0]
                
                if len(valid_dofs) > 0:
                    # Find DOF closest to the target z-coordinate
                    z_distances = np.abs(bc_dof_coords[valid_dofs, 0] - z_coord)
                    closest_idx = valid_dofs[np.argmin(z_distances)]
                    dof_idx = bc_dofs[closest_idx]
                    
                    watcher_nodes.append(dof_idx)
                    watcher_use_bc_value.append(True)
                    watcher_bc_funcs.append(heating_bc_obj._value_callable)
                    
                    # Diagnostic output
                    dof_coord = bc_dof_coords[closest_idx]
                    print(f"Watcher '{name}' at heating_location: using exact boundary DOF {dof_idx}")
                    print(f"  Target: z={z_coord:.6e}, r={r_coord:.6e}")
                    print(f"  DOF:    z={dof_coord[0]:.6e}, r={dof_coord[1]:.6e}")
                    print(f"  Distance: z={abs(dof_coord[0]-z_coord):.2e}, r={abs(dof_coord[1]-r_coord):.2e}")
                else:
                    # Fallback to nearest mesh node
                    distance, node_idx = tree.query(coords)
                    watcher_nodes.append(node_idx)
                    watcher_use_bc_value.append(False)
                    watcher_bc_funcs.append(None)
                    print(f"Watcher '{name}' at heating_location: no boundary DOF found at r={r_coord:.6e}, using nearest node {node_idx}")
            else:
                # Standard case: find nearest mesh node
                distance, node_idx = tree.query(coords)
                watcher_nodes.append(node_idx)
                watcher_use_bc_value.append(False)
                watcher_bc_funcs.append(None)
        
        return {
            'names': watcher_names,
            'nodes': watcher_nodes,
            'use_bc_value': watcher_use_bc_value,  # Whether to record BC value directly
            'bc_funcs': watcher_bc_funcs,  # BC functions for direct value recording
            'data': {name: [] for name in watcher_names},
            'time': [],
            'filename': watcher_cfg.get('filename', 'watcher_points.csv')
        }
    
    def _run_time_stepping(self, domain, V, lhs_form, rhs_form, obj_bcs, bcs, solver, 
                          u_n, xdmf_file, watcher_setup):
        """Run the time stepping loop with optimizations."""
        t_final = float(self.cfg['timing']['t_final'])
        num_steps = int(self.cfg['timing']['num_steps'])
        dt = t_final / num_steps
        
        # Initialize vectors - Fix: Use correct DOLFINx import
        from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc, create_vector
        b = create_vector(rhs_form)
        
        # Progress reporting
        progress_cfg = self.cfg.get('performance', {}).get('progress_reporting', {})
        progress_enabled = progress_cfg.get('enabled', True)
        progress_interval = progress_cfg.get('interval', 5)
        progress_steps = max(1, num_steps // (100 // progress_interval))
        
        step_times = []
        loop_start_time = time.time()
        
        # Initialize boundary conditions
        for bc in obj_bcs:
            bc.update(0.0)
        
        # Write initial condition
        if xdmf_file:
            xdmf_file.write_function(u_n, 0.0)
        
        for step in range(num_steps):
            step_start = time.time()
            t = (step + 1) * dt
            
            # Update boundary conditions
            for bc in obj_bcs:
                bc.update(t)
            
            # Assemble and solve
            with b.localForm() as local_b:
                local_b.set(0)
            assemble_vector(b, rhs_form)
            apply_lifting(b, [lhs_form], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)
            solver.solve(b, u_n.x.petsc_vec)
            u_n.x.scatter_forward()
            
            # Write output
            if xdmf_file:
                xdmf_file.write_function(u_n, t)
            
            # Record watcher point data
            if watcher_setup:
                watcher_setup['time'].append(t)
                for i, (name, node_idx) in enumerate(zip(watcher_setup['names'], watcher_setup['nodes'])):
                    use_bc_value = watcher_setup.get('use_bc_value', [False] * len(watcher_setup['names']))[i]
                    bc_func = watcher_setup.get('bc_funcs', [None] * len(watcher_setup['names']))[i]
                    
                    # Always record solution value (even for heating_location watchers)
                    # The solution value at a Dirichlet BC DOF should equal the BC value
                    try:
                        val = u_n.x.array[node_idx]
                        
                        # Diagnostic: compare BC value vs solution value for heating_location watchers
                        if use_bc_value and bc_func is not None and step == 0 and i == 0:
                            try:
                                from dolfinx import fem
                                dof_coords = V.tabulate_dof_coordinates()
                                dof_coord = dof_coords[node_idx]
                                z_coord = dof_coord[0]
                                r_coord = 0.0
                                bc_val = bc_func(z_coord, r_coord, t)
                                
                                # Get experimental data for comparison
                                heating_data = self._load_heating_data()
                                exp_times = heating_data['time'].values
                                exp_temps = heating_data['temp'].values
                                
                                # Find closest experimental point
                                exp_idx = np.argmin(np.abs(exp_times - t))
                                exp_temp_at_t = exp_temps[exp_idx]
                                exp_time_at_t = exp_times[exp_idx]
                                
                                # Get baseline and offset for comparison
                                baseline_temp = self._compute_baseline(exp_times, exp_temps)
                                ic_temp = float(self.cfg['heating']['ic_temp'])
                                offset = baseline_temp - ic_temp
                                
                                print(f"Diagnostic (step {step}, t={t:.6e}):")
                                print(f"  Solution value:      {val:.6f} K")
                                print(f"  BC value at r=0:     {bc_val:.6f} K")
                                print(f"  Difference:          {abs(bc_val - val):.6e} K")
                                print(f"  Exp temp at t={exp_time_at_t:.6e}: {exp_temp_at_t:.6f} K")
                                print(f"  Baseline:            {baseline_temp:.6f} K")
                                print(f"  IC temp:              {ic_temp:.6f} K")
                                print(f"  Offset:               {offset:.6f} K")
                                print(f"  BC formula:          exp_temp - offset = {exp_temp_at_t:.6f} - {offset:.6f} = {exp_temp_at_t - offset:.6f} K")
                                print(f"  Exp max:              {exp_temps.max():.6f} K")
                                print(f"  Exp min:              {exp_temps.min():.6f} K")
                            except Exception as e:
                                print(f"Diagnostic error: {e}")
                    except:
                        val = np.nan
                    watcher_setup['data'][name].append(val)
            
            # Progress reporting
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            if progress_enabled and (step + 1) % progress_steps == 0:
                percent = int((step + 1) / num_steps * 100)
                avg_step_time = np.mean(step_times[-progress_steps:])
                print(f"Progress: {percent}% | Avg step time: {avg_step_time:.4f}s")
        
        # Calculate timing statistics
        total_loop_time = time.time() - loop_start_time
        avg_step_time = np.mean(step_times) if step_times else 0.0
        
        return {
            'total_loop_time': total_loop_time,
            'avg_step_time': avg_step_time,
            'num_steps': num_steps
        }
    
    def _save_watcher_data(self, watcher_setup):
        """Save watcher data to CSV."""
        if not watcher_setup or not watcher_setup['data']:
            return
        
        csv_path = os.path.join(self.output_folder, watcher_setup['filename'])
        
        df = pd.DataFrame({'time': watcher_setup['time']})
        for name in watcher_setup['names']:
            df[name] = watcher_setup['data'][name]
        df.to_csv(csv_path, index=False)

    def _visualize_mesh(self):
        """Visualize the mesh using gmsh."""
        try:
            import gmsh
            
            mesh_file_path = os.path.join(self.mesh_folder, 'mesh.msh')
            if not os.path.exists(mesh_file_path):
                print(f"Warning: Mesh file not found at {mesh_file_path}")
                return
            
            print("Visualizing mesh...")
            gmsh.initialize()
            gmsh.open(mesh_file_path)
            gmsh.fltk.initialize()
            gmsh.fltk.run()
            gmsh.finalize()
            print("Mesh visualization closed.")
            
        except ImportError:
            print("Warning: gmsh not available for mesh visualization")
        except Exception as e:
            print(f"Error visualizing mesh: {e}")

    def run_minimal(self, suppress_print=False):
        """
        Run a minimal simulation with no disk I/O, returning only QoIs.
        
        Parameters:
        -----------
        suppress_print : bool, optional
            If True, suppress all print output
        
        Returns:
        --------
        dict
            Dictionary containing:
            - 'watcher_data': dict with time series and normalized curves
            - 'config': the configuration used for the simulation
            - 'timing': timing information
        """
        with suppress_output(suppress_print):
            try:
                print("[DEBUG] Starting minimal heat flow simulation...")
                # Build mesh in memory (no disk I/O)
                print("[DEBUG] Building mesh in memory...")
                domain, cell_tags, materials = self._build_mesh_in_memory()
                print("[DEBUG] Mesh built.")
                # Setup function spaces and material properties
                print("[DEBUG] Setting up function spaces...")
                V, Q, kappa, rho_cv = self._setup_function_spaces(domain, cell_tags, materials)
                print("[DEBUG] Function spaces set up.")
                # Load heating data
                print("[DEBUG] Loading heating data...")
                heating_data = self._load_heating_data()
                print("[DEBUG] Heating data loaded.")
                # Setup boundary conditions
                print("[DEBUG] Setting up boundary conditions...")
                obj_bcs, bcs, heating_bc_obj = self._setup_boundary_conditions(V, materials, heating_data)
                print("[DEBUG] Boundary conditions set up.")
                # Initialize solution
                print("[DEBUG] Initializing solution...")
                u_n = fem.Function(V)
                ic_temp = float(self.cfg['heating']['ic_temp'])
                u_n.x.array[:] = np.full_like(u_n.x.array, ic_temp)
                u_n.x.scatter_forward()
                u_n.name = 'Temperature (K)'
                print("[DEBUG] Solution initialized.")
                # Setup variational forms
                print("[DEBUG] Setting up variational forms...")
                lhs_form, rhs_form = self._setup_variational_forms(domain, V, Q, kappa, rho_cv, u_n)
                print("[DEBUG] Variational forms set up.")
                # Setup solver
                print("[DEBUG] Setting up solver...")
                solver = self._setup_solver(lhs_form, bcs)
                print("[DEBUG] Solver set up.")
                # Setup watcher points only (no XDMF)
                print("[DEBUG] Setting up watcher points...")
                watcher_setup = self._setup_watcher_points_minimal(domain, heating_bc_obj)
                print("[DEBUG] Watcher points set up.")
                # Run time stepping
                print("[DEBUG] Running time stepping...")
                timing_results = self._run_time_stepping_minimal(
                    domain, V, lhs_form, rhs_form, obj_bcs, bcs, solver, u_n, watcher_setup
                )
                print("[DEBUG] Time stepping complete.")
                # Process watcher data to get raw and normalized curves
                print("[DEBUG] Processing watcher data...")
                watcher_data = self._process_watcher_data(watcher_setup)
                print("[DEBUG] Watcher data processed.")
                return {
                    'watcher_data': watcher_data,
                    'config': self.cfg,
                    'timing': timing_results
                }
            except Exception as e:
                print("[ERROR] Exception in run_minimal:", e)
                traceback.print_exc()
                raise

    def _build_mesh_in_memory(self):
        """Build mesh in memory without saving to disk."""
        # Create materials from configuration
        materials = self._create_materials_from_config()
        
        # Calculate mesh boundaries
        mesh_bounds = self._calculate_mesh_boundaries(materials)
        
        # Create mesh
        gmsh_domain = Mesh(
            name='mesh.msh',
            boundaries=mesh_bounds,
            materials=list(materials.values())
        )
        
        # Build mesh in memory (don't save to disk)
        gmsh_domain.build_mesh()
        
        # Convert to DOLFINx directly from memory using the Mesh class method
        domain, cell_tags, facet_tags = gmsh_domain.to_dolfinx()
        
        return domain, cell_tags, materials

    def _setup_watcher_points_minimal(self, domain, heating_bc_obj=None):
        """Setup watcher points for minimal run (no file output)."""
        output_cfg = self.cfg.get('output', {})
        
        if not output_cfg.get('watcher_points', {}).get('enabled', True):
            return None
        
        return self._setup_watcher_points(domain, output_cfg['watcher_points'], heating_bc_obj)

    def _run_time_stepping_minimal(self, domain, V, lhs_form, rhs_form, obj_bcs, bcs, solver, 
                                  u_n, watcher_setup):
        """Run time stepping for minimal run (no XDMF output)."""
        t_final = float(self.cfg['timing']['t_final'])
        num_steps = int(self.cfg['timing']['num_steps'])
        dt = t_final / num_steps
        
        # Initialize vectors
        from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc, create_vector
        b = create_vector(rhs_form)
        
        # Disable progress reporting for minimal runs
        step_times = []
        loop_start_time = time.time()
        
        # Initialize boundary conditions
        for bc in obj_bcs:
            bc.update(0.0)
        
        for step in range(num_steps):
            step_start = time.time()
            t = (step + 1) * dt
            
            # Update boundary conditions
            for bc in obj_bcs:
                bc.update(t)
            
            # Assemble and solve
            with b.localForm() as local_b:
                local_b.set(0)
            assemble_vector(b, rhs_form)
            apply_lifting(b, [lhs_form], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)
            solver.solve(b, u_n.x.petsc_vec)
            u_n.x.scatter_forward()
            
            # Record watcher point data only
            if watcher_setup:
                watcher_setup['time'].append(t)
                for i, (name, node_idx) in enumerate(zip(watcher_setup['names'], watcher_setup['nodes'])):
                    use_bc_value = watcher_setup.get('use_bc_value', [False] * len(watcher_setup['names']))[i]
                    bc_func = watcher_setup.get('bc_funcs', [None] * len(watcher_setup['names']))[i]
                    
                    # Always record solution value (even for heating_location watchers)
                    # The solution value at a Dirichlet BC DOF should equal the BC value
                    try:
                        val = u_n.x.array[node_idx]
                    except:
                        val = np.nan
                    watcher_setup['data'][name].append(val)
            
            step_time = time.time() - step_start
            step_times.append(step_time)
        
        # Calculate timing statistics
        total_loop_time = time.time() - loop_start_time
        avg_step_time = np.mean(step_times) if step_times else 0.0
        
        return {
            'total_loop_time': total_loop_time,
            'avg_step_time': avg_step_time,
            'num_steps': num_steps
        }

    def _process_watcher_data(self, watcher_setup):
        """Process watcher data to extract raw and normalized temperature curves."""
        if not watcher_setup or not watcher_setup['data']:
            return {}
        
        # Convert to numpy arrays
        time_array = np.array(watcher_setup['time'])
        watcher_data = {}
        
        # First pass: collect raw data and calculate pside baseline & excursion
        baseline_cfg = self.cfg.get('baseline', {})
        use_avg = bool(baseline_cfg.get('use_average', False))
        t_window = float(baseline_cfg.get('time_window', 0.0))

        pside_data = None
        pside_baseline = None
        pside_max_excursion = None

        for name in watcher_setup['names']:
            raw_data = np.array(watcher_setup['data'][name])
            
            if name == 'pside':
                # Compute baseline for pside
                mask = time_array <= t_window if use_avg else np.zeros_like(time_array, dtype=bool)
                if not use_avg:
                    mask[0] = True  # Ensure at least first sample included
                elif not mask.any():
                    # Fallback to first datapoint if window too small
                    mask[0] = True

                pside_baseline = np.mean(raw_data[mask])
                pside_data = raw_data  # Store for later normalization logic

                # Excursion relative to baseline
                pside_max_excursion = (raw_data - pside_baseline).max() - (raw_data - pside_baseline).min()

        # Second pass: process all watcher points
        for name in watcher_setup['names']:
            raw_data = np.array(watcher_setup['data'][name])

            # Compute baseline for this watcher (may be pside or others)
            mask = time_array <= t_window if use_avg else np.zeros_like(time_array, dtype=bool)
            if not use_avg or not mask.any():
                mask = np.zeros_like(time_array, dtype=bool)
                mask[0] = True

            baseline_val = np.mean(raw_data[mask])
            
            if name == 'pside' and pside_data is not None:
                # P-side: subtract its own baseline, scale by its excursion
                if pside_max_excursion > 0:
                    normalized_data = (raw_data - pside_baseline) / pside_max_excursion
                else:
                    normalized_data = np.zeros_like(raw_data)
                max_excursion = pside_max_excursion

            elif name == 'oside' and pside_data is not None:
                # O-side: subtract its *own* baseline but still scale by p-side excursion
                if pside_max_excursion > 0:
                    normalized_data = (raw_data - baseline_val) / pside_max_excursion
                else:
                    normalized_data = np.zeros_like(raw_data)
                max_excursion = pside_max_excursion
            
            else:
                # Normalize other points relative to their own excursion
                shifted = raw_data - baseline_val
                max_excursion = shifted.max() - shifted.min()
                if max_excursion > 0:
                    normalized_data = shifted / max_excursion
                else:
                    normalized_data = np.zeros_like(raw_data)
            
            watcher_data[name] = {
                'time': time_array,
                'raw': raw_data,
                'normalized': normalized_data,
                'initial_temp': baseline_val,
                'max_excursion': max_excursion
            }
        
        return watcher_data

    def _compute_baseline(self, time_series: np.ndarray, temp_series: np.ndarray) -> float:
        """Compute baseline temperature using either the first data point or an
        average over an initial time window, depending on *cfg['baseline']*.

        If *cfg['baseline']['use_average']* is True, the baseline is taken as
        the mean of *temp_series* for which *time_series <= time_window* where
        *time_window* comes from *cfg['baseline']['time_window']* (defaults to
        0).  If the mask is empty (e.g. the window is smaller than the first
        measurement interval) we fall back to the first data point so the
        behaviour is robust.
        """
        baseline_cfg = self.cfg.get('baseline', {})
        use_avg = bool(baseline_cfg.get('use_average', False))
        if not use_avg:
            return float(temp_series[0])

        # Time window defaults to 0 so that, if unspecified, we again take the
        # first data point (same as old behaviour).
        t_window = float(baseline_cfg.get('time_window', 0.0))

        mask = time_series <= t_window
        if mask.any():
            return float(np.mean(temp_series[mask]))
        # Fallback – no points in window.
        return float(temp_series[0])


# Alias for backward compatibility
SimulationEngine = OptimizedSimulationEngine
