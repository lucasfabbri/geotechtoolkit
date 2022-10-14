import pandas as pd
import pyvista as pv
import gstools as gs
import numpy as np
import random

from shapely.geometry import Point

from utils import get_point_on_vector, get_length


def simulate_intersection_feature(investigation, surface, description):
    intersections = investigation.z_coords_on_surface(surface)
    description_array = [description for _ in range(len(intersections[:, 0]))]
    return pd.DataFrame(
        {
            "ID": intersections[:, 0],
            "Z_feature": intersections[:, 1],
            "description": description_array,
        }
    )


def random_field(grid, inplace=True, seed=False, **kwargs):
    """Generate a random field over a grid

    Args:
        grid (pyvista.GriddedData): Grid over which to generate random field
        inplace (bool): Specifiy if the random field should be added to the grid or returned as a numpy array
        seed (int): Seed for gstools model
        **kwargs (dict): Keyword arguments for the gstools.Gaussian method

    """
    if not inplace:
        grid = grid.copy()

    model = gs.Gaussian(**kwargs)
    srf = gs.SRF(model, seed=seed)
    srf.mesh(grid, points="points", name="random_field")

    return np.asarray(grid["random_field"])


def generate_random_even_spaced_layers(
    n_layers, top_surface, bottom_surface, seed=False, edge_spacing=10
):

    maximum = top_surface.points[:, 2].max() - edge_spacing
    minimum = bottom_surface.points[:, 2].min() + edge_spacing

    x = np.arange(bottom_surface.bounds[0], bottom_surface.bounds[1], 1)
    y = np.arange(bottom_surface.bounds[2], bottom_surface.bounds[3], 1)

    boundaries = np.linspace(minimum, maximum, n_layers)

    if not seed:
        seed = random.random((0, 1000000))

    surfs = []

    for i, boundary in enumerate(boundaries):
        model = gs.Gaussian(dim=2, var=4, len_scale=10)
        srf = gs.SRF(model, seed=(seed + i))
        srf((x, y), mesh_type="structured")
        z = np.concatenate(srf.field)

        z = z + boundary

        array = []
        j = 0
        for x_i in x:
            for y_i in y:
                array.append((x_i, y_i, z[j]))
                j += 1

        poly = pv.PolyData(np.asarray(array))
        surf = poly.delaunay_2d()
        surfs.append(surf)

    trimmed = []

    for surf in surfs:
        rock_trimmed = surf.clip_surface(bottom_surface, invert=False)
        topo_trimmed = rock_trimmed.clip_surface(top_surface, invert=True)
        trimmed.append(topo_trimmed)

    return sorted(trimmed, key=lambda x: x.center_of_mass()[-1])


def apply_random_field_to_properties(grid, properties):
    keys = properties.keys()
    random_field = np.asarray(grid["random_field"])
    unit_index = np.asarray(grid["unit_index"])
    cohesion_sd = np.copy(unit_index)
    cohesion_mean = np.copy(unit_index)
    friction_sd = np.copy(unit_index)
    friction_mean = np.copy(unit_index)
    for key in keys:
        cohesion_sd[cohesion_sd == key] = properties[key]["cohesion"][1]
        cohesion_mean[cohesion_mean == key] = properties[key]["cohesion"][0]
        friction_sd[friction_sd == key] = properties[key]["friction_angle"][1]
        friction_mean[friction_mean == key] = properties[key]["friction_angle"][0]
    cohesion = (cohesion_sd * random_field) + cohesion_mean
    cohesion[cohesion < 0] = 0

    friction = (friction_sd * random_field) + friction_mean
    friction[friction < 0] = 0

    grid["cohesion"] = cohesion
    grid["friction"] = friction


def sample_over_line(grid, collar, resolution=0.02):
    """Wrapper to use pyvistas sample_over_line and return as a dataframe with the Investigation.data format

    Args:
        grid (pyvista.GriddedData): Grid with data to sample
        collar (pandas.DataFrame):

    """
    ID = collar["ID"].values[0]
    TYPE = "pv.sample_over_line"
    point = (collar.X.values[0], collar.Y.values[0], collar.Z.values[0])
    min_point = (collar.X.values[0], collar.Y.values[0], grid.bounds[2])

    length = get_length(point, min_point)

    sample_res = int(length / resolution)

    sampled = grid.sample_over_line(point, min_point, resolution=sample_res)
    keys = grid.point_data.keys()
    pts = [
        Point(get_point_on_vector(point, min_point, round(distance, 3)))
        for distance in sampled["Distance"]
    ]
    data = {
        "ID": [ID] * len(pts),
        "Point": pts,
        "Type": [TYPE] * len(pts),
        "InVolume": sampled["vtkValidPointMask"],
    }

    for key in keys:
        data[key] = sampled[key]

    df = pd.DataFrame(data)
    df = df[df["InVolume"] == 1]
    del df["InVolume"]
    return df
