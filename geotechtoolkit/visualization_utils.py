from turtle import circle
import pyvista as pv
import numpy as np
from shapely.geometry import LineString


from .consts import FEATURE_STYLES


def collar_cone(point, direction=(0, 0, -1)):

    x = point[0]
    y = point[1]
    z = point[2]

    return pv.Cone(center=(x, y, z + 0.5), direction=(0, 0, -1), height=1, resolution=6)


def feature_disc(point, normal):
    return pv.Disc(point, inner=0, outer=0.5, normal=normal, c_res=10)


def plot_collars(investigation, plotter=False):
    """Plot collars as black inverted triangles with label as text
    Input is investigation object

    If plotter is given it will return given plotter, if not it will return dictionary of actors
    """
    return_actors = False
    ids = investigation.collars["ID"]
    X = investigation.collars["X"]
    Y = investigation.collars["Y"]
    Z = investigation.collars["Z"]

    if not plotter:
        plotter = pv.Plotter()
        return_actors = True
    actors = {}

    for id, x, y, z in zip(ids, X, Y, Z):

        text = pv.Text3D(id, depth=0.25)
        transform_matrix = np.array(
            [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
        )
        text.transform(transform_matrix)
        symbol = collar_cone((x, y, z))
        actors[str(id)] = {}
        text_actor = plotter.add_mesh(text, color="black")
        cone_actor = plotter.add_mesh(symbol, color="black")
        actors[id]["text"] = text_actor
        actors[id]["symbol"] = cone_actor

    return actors if return_actors else plotter


def plot_features(investigation, plotter=False, styles=FEATURE_STYLES):
    return_actors = False

    ids = investigation.features["ID"]

    if not plotter:
        plotter = pv.Plotter()
        return_actors = True

    actors = {}
    for id in ids:
        actors[id] = []
        collar = investigation.collars.loc[investigation.collars["ID"] == id]
        features = investigation.features.loc[investigation.features["ID"] == id]
        for feature in features.itertuples():
            circle = pv.Disc(**styles[feature.description]["disc"])
            transform_matrix = np.array(
                [
                    [1, 0, 0, float(collar["X"])],
                    [0, 1, 0, float(collar["Y"])],
                    [0, 0, 1, float(feature.Z_feature)],
                    [0, 0, 0, 1],
                ]
            )
            circle.transform(transform_matrix)
            temp_actor = plotter.add_mesh(circle, **styles[feature.description]["plot"])
            actors[id].append(temp_actor)

    return actors if return_actors else plotter


def investigation_data_to_polydata(investigation):
    inv_data = investigation.data
    linestring = LineString(list(inv_data["Point"].values))
    polydata = pv.PolyData(
        np.asarray([[coord[0], coord[1], coord[2]] for coord in linestring.coords])
    )
    columns = list(inv_data.columns)
    for key in columns:
        if key not in ["ID", "Point", "Type"]:
            polydata[key] = np.asarray(inv_data[key])
    return polydata
