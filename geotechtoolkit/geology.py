import numpy as np
import pyvista as pv
from pykrige.uk import UniversalKriging

def get_x_y_bounds(points):
    """ Get the maximum and minimum of x and y coordinates
    Args:
        points(numpy.array): Numpy array of points
    Returns:
        xmin (float): Minimum x coordinate
        xmax (float): Maximum x coordinate
        ymin (float): Minimum y coordinate
        ymax (float): Maximum y coordinate
    """

    xmin = float(np.min(points[:,0]))
    xmax = float(np.max(points[:,0]))
    ymin = float(np.min(points[:,1]))
    ymax = float(np.max(points[:,1]))

    return xmin,xmax,ymin,ymax

class GeologyElement():
    """ Base class for all geolgical elements containing metadata

    Attributes:
        name (str): Name of element
        comment (str): Relevant comment for element
    
    """
    def __init__(self,name,comment):
        self.name = name
        self.comment = comment

class Interpolation():
    """ Class to store interpolated surface 

    Contains scalar field representing uncetainty

    Attributes:
        parameters (dict): Dictionary with parameters used to generate interpolation
        surface (PyVista Gridded Data): Grid representing the interpolated surface

    """
<<<<<<< Updated upstream
    def __init__(self,parameters,surface):
        self.parameters = parameters
        self.surface = surface    
=======

    def __init__(self, metadata, surface):
        self.metadata = metadata
        self.surface = surface

    def mean_squared_error(self, real_surface):
        from sklearn.metrics import mean_squared_error

        X_predicted = np.asarray(self.surface.points[:, 0])
        Y_predicted = np.asarray(self.surface.points[:, 1])
        Z_predicted = np.asarray(self.surface.points[:, 2])
        ones = np.ones(len(X_predicted))
        Z_max = ones * MAX_ELEVATION
        Z_min = ones * MIN_ELEVATION
        Z_expected = np.asarray(
            [
                real_surface.ray_trace([x, y, z_max], [x, y, z_min])[0][0][-1]
                for x, y, z_max, z_min in zip(X_predicted, Y_predicted, Z_max, Z_min)
            ]
        )

        return mean_squared_error(Z_expected, Z_predicted)

    def two_tailed_test(self, real_surface):
        from scipy.stats import norm

        X_predicted = np.asarray(self.surface.points[:, 0])
        Y_predicted = np.asarray(self.surface.points[:, 1])
        Z_predicted = np.asarray(self.surface.points[:, 2])

        ones = np.ones(len(X_predicted))
        Z_max = ones * MAX_ELEVATION
        Z_min = ones * MIN_ELEVATION
        Z_expected = np.asarray(
            [
                real_surface.ray_trace([x, y, z_max], [x, y, z_min])[0][0][-1]
                for x, y, z_max, z_min in zip(X_predicted, Y_predicted, Z_max, Z_min)
            ]
        )

        standard_deviation = np.sqrt(self.metadata["uncertainty"]["variance"])

        z_score = (Z_expected - Z_predicted) / standard_deviation

        return norm.sf(abs(z_score)) * 2


def compute_distance_to_points(surface, original_points):
    distances = []
    for surfpt in surface.points:
        localdistances = []
        for origpt in original_points:
            sfpt_to_orpt = (
                (origpt[0] - surfpt[0]) ** 2
                + (origpt[1] - surfpt[1]) ** 2
                + (origpt[2] - surfpt[2]) ** 2
            ) ** 0.5
            localdistances.append(sfpt_to_orpt)
        distances.append(min(localdistances))
    return distances

>>>>>>> Stashed changes

class Surface(GeologyElement):
    """Geological surfaces without holes, i.e open surfaces.

    Can represent simplified boundaries between strata, bedrock surfaces, discontinuities etc.

    Attributes: 
        data_source_type (str): Descriptor of what kind of data is the source for the surface.
<<<<<<< Updated upstream
        interpolation (Interpolation): Interpolation subclass
            parameters (dict): Metadata about the interpolation method used
            surface (PyVista Gridded Data): Grid representing the interpolated surface
    """
    def __init__(self,name,comment, data_souce_type):
        super().__init__(name,comment)
=======
        surface (pyvista.PolyData): PyVista object representing the surface
    """

    def __init__(self, name, surface=False, comment=None, data_souce_type=None):
        super().__init__(name, comment)
>>>>>>> Stashed changes
        self.data_souce_type = data_souce_type
        if surface:
            self.surface = surface
            self.interpolation = False

    def triangulate(self,points):

        """ Delaunay 2D triangulation using pyvista
            
            Args:
                points (np.array): Numpy array of points

            Attributes:

        """
        parameters = {"type":"triangulate","params":{"points":points}}
        if len(points) < 3:
            # TODO: plane that connects 2 points or is planar at 1 z coord
            print("Cannot interpolate surface from two points")
        elif len(points) == 3:
            verticies = points
            faces = np.hstack([3, 0, 1, 2])
            surf = pv.PolyData(verticies, faces)
        else:
            cloud = pv.PolyData(points)
<<<<<<< Updated upstream
            surf = cloud.delaunay_2d()

        #TODO: Uncertainty in triangles based on (?)

        self.interpolation = Interpolation(parameters,surf)
 
    def krige(self,
            points,
            resolution = 1,
            nlags = 8,
            weight = True,
            variogram_model = "gaussian",
            exact_values=True,
            drift_terms = ["regional_linear"],
            exrapolation_buffer = (0,0,0),
            boundary_polyline=False,
            ):
        """ Interpolate by kriging using PyKriges UniversialKriging method       
=======
            surface = cloud.delaunay_2d()
        metadata = {
            "type": "triangulate",
            "params": {"points": points},
            "uncertainty": {"distance": compute_distance_to_points(surface, points)},
        }

        # TODO: Uncertainty in triangles based on (?)

        self.interpolation = Interpolation(metadata, surface)
        self.surface = surface

    def krige(
        self,
        points,
        resolution=1,
        nlags=6,
        weight=True,
        variogram_model="linear",
        exact_values=True,
        drift_terms=None,
        extrapolation_buffer=EXTRAPOLATION_BUFFER,
        domain=None,
    ):
        """Interpolate by kriging using PyKriges UniversialKriging method
>>>>>>> Stashed changes
        Overwrites existing interpolation for the model

        Args:
            points (np.array): Array of points
            resoloution (float or int):Grid x,y resolution to interpolate in
            nlags (int): Number of lags for the kriging variogram
            weight (bool): Specify if the lags should be weighted #TODO: Find wording from PyKrige
            variogram_model (str): Variogram model to be used
            drift_terms (str): Drift terms to be used
            extrapolation_buffer (tuple): Distance to extrapolate in (x,y,z)
            boundary_polyline (optional,LineString): Shapely LineString or similar format defining outer
                                            boundary to clip surface against. Defaults to False
        """
        
        UK = UniversalKriging(
            points[:,0],
            points[:,1],
            points[:,2],
            variogram_model=variogram_model,
            nlags = nlags,
            weight=weight,
            drift_terms=drift_terms,
            exact_values=exact_values
        )

        xmin,xmax,ymin,ymax = get_x_y_bounds(points)

        X = np.arange(xmin-exrapolation_buffer[0],xmax + exrapolation_buffer[0],resolution)
        Y = np.arange(ymin-exrapolation_buffer[1],ymax+exrapolation_buffer[1],resolution)
        print(xmin,xmax,ymin,ymax)

        Z, ss = UK.execute("grid",X,Y)
        Z = np.squeeze(np.array(Z))

        
        print(X.size,Y.size,Z.size)
        
        #TODO: Use structured grid grid = pv.StructuredGrid(X, Y, Z)

        UK_xyz = []
        for i_y, row in enumerate(Z):
            for i_x, zval in enumerate(row):
                UK_xyz.append([X[i_x], Y[i_y], zval])

        

        UK_xyz = np.asarray(UK_xyz)

        prediction = pv.PolyData(UK_xyz)
<<<<<<< Updated upstream
        print("Meshing kriged grid")
        grid = prediction.delaunay_2d()
        print("Mesh created")
=======
        surface = prediction.delaunay_2d()
        metadata = {
            "type": "krige",
            "params": {
                "points": points,
                "resolution": resolution,
                "nlags": nlags,
                "weight": weight,
                "variogram_model": variogram_model,
                "exact_values": exact_values,
                "drift_terms": drift_terms,
            },
            "uncertainty": {
                "variance": np.asarray(ss).flatten(),
                "distance": compute_distance_to_points(surface, points),
            },
            "UK": UK,
        }

        self.interpolation = Interpolation(metadata, surface)
        self.surface = surface

    def add_uncertainty_visualization(self):
        uncertainty_rep_types = self.interpolation.metadata["uncertainty"].keys()
        for type in uncertainty_rep_types:
            self.surface[type] = np.asarray(
                self.interpolation.metadata["uncertainty"][type]
            )
>>>>>>> Stashed changes

        parameters = {
            "type":"krige",
            "params":{
                "points":points,
                "resolution":resolution,
                "nlags":nlags,
                "weight":weight,
                "variogram_model":variogram_model,
                "exact_values":exact_values,
                "drift_terms":drift_terms,
                
            }
        }

<<<<<<< Updated upstream
#        if boundary_polyline:
#            #TODO: clip by surface based on extruded surface from polyline StructuredGrid.clip_surface(surface[, ...])
        
        #TODO: Uncertainty based on
        self.interpolation = Interpolation(parameters,grid)
=======
class GeologicalModel:
    """Class containing all interpreted geological elements

    # TODO: 3D model of domain bounded by top and bottom

    Attributes:
    """

    def __init__(
        self,
        points=None,
        zMax=None,
        zMin=None,
        extrapolation_buffer=EXTRAPOLATION_BUFFER,
        domain=None,
    ):
        """Initialize an area based on either a set of coordinates an an extrapolation buffer, or specify a domain directly.
        If domain is specified, other inputs are ignored.
        domain can be specified directly = (xMin, xMax, yMin, yMax, zMin, zMax).

        Args:
            points (np.array): Numpy array of points [(x1,y1,z1),...,(xn,yn,zn)]
            extrapolation_buffer (tuple): Extra area to interpolate outside of boundary of points
            domain = (xMin, xMax, yMin, yMax, zMin, zMax).
        """
        if domain:
            self.xMin = round(domain[0], 2)
            self.xMax = round(domain[1], 2)
            self.yMin = round(domain[2], 2)
            self.yMax = round(domain[3], 2)
            self.zMin = round(domain[4], 2)
            self.zMax = round(domain[5], 2)
        else:
            xmin, xmax, ymin, ymax = get_x_y_bounds(points)
            if len(points[0]) == 3:
                zmax = float(np.max(points[:, 2]))
                zmin = float(np.min(points[:, 2]))

            self.xMin = round(xmin - extrapolation_buffer[0], 2)
            self.xMax = round(xmax + extrapolation_buffer[0], 2)
            self.yMin = round(ymin - extrapolation_buffer[1], 2)
            self.yMax = round(ymax + extrapolation_buffer[1], 2)
            self.zMin = round(zmin - extrapolation_buffer[2], 2)
            self.zMax = round(zmax + extrapolation_buffer[2], 2)
        if zMax:
            self.zMax = zMax
        if zMin:
            self.zMin = zMin
        self.surfaces = []

    @property
    def domain(self):
        return (self.xMin, self.xMax, self.yMin, self.yMax, self.zMin, self.zMax)

    def add_topography(self, topography, name="Topography"):
        topography = Surface(name, surface=topography)
        self.topography = topography

    def add_basement(self, basement, name="Basement"):
        basement = Surface(name, surface=basement)
        self.basement = basement

    def add_surface(self, surface, name=None, **kwargs):
        """Add Surface object directly"""
        surface = Surface(name=name, surface=surface, **kwargs)
        self.surfaces.append(surface)

    def discretize(self, resolution=1):
        """Create a rectilinear grid over the specified model domain.

        If the object has a basement or topography, the grid is constrained by these surfaces

        """
        xx = np.arange(self.xMin, self.xMax, resolution)
        yy = np.arange(self.yMin, self.yMax, resolution)
        zz = np.arange(self.zMin, self.zMax, resolution)
        self.grid = pv.RectilinearGrid(xx, yy, zz)

        if hasattr(self, "basement"):
            self.grid = self.grid.clip_surface(self.basement.surface, invert=False)
        if hasattr(self, "topography"):
            self.grid = self.grid.clip_surface(self.topography.surface)

    def divide_by_surfaces(self):
        """Using implicit distances, subdivide the grid into zones deliniated by the GeologicalModel's surfaces

        Creates a new scalar on the grid called "unit_index" with ascending values for each layer by depth
        (i.e shallowest layer = 1, next = 2, ... deepest = n)

        """
        arrays = []
        surfs = [surface.surface for surface in self.surfaces]
        trimmed = sorted(surfs, key=lambda x: x.center_of_mass()[-1], reverse=True)

        for i in range(len(trimmed) + 1):
            if i == 0:
                filterer = self.grid.compute_implicit_distance(trimmed[0])
                array = np.asarray(filterer["implicit_distance"])
                array[array < 0] = 0
                array[array > 0] = i + 1
            elif i == len(trimmed):
                filterer = self.grid.compute_implicit_distance(trimmed[-1])
                array = np.asarray(filterer["implicit_distance"])
                array[array > 0] = 0
                array[array < 0] = i + 1
            else:
                filterer = self.grid.compute_implicit_distance(trimmed[i])
                array1 = np.asarray(filterer["implicit_distance"])
                array1[array1 < 0] = 0
                array1[array1 > 0] = i + 1
                filterer = self.grid.compute_implicit_distance(trimmed[i - 1])
                array2 = np.asarray(filterer["implicit_distance"])
                array2[array2 > 0] = 0
                array2[array2 < 0] = i + 1
                array = array1 * array2
                array[array < 0] = 0
                array[array > 0] = i + 1

            arrays.append(array)
        self.grid["unit_index"] = sum(arrays)
        return arrays
>>>>>>> Stashed changes
