""" https://help.seequent.com/Geo/4.4/en-GB/Content/drillholes/import-drillholes.htm"""
import numpy as np
import pandas as pd
import pyvista as pv
from visualization_utils import plot_collars, plot_features

from geology import Surface, GeologicalModel
from CONSTS import MAX_ELEVATION, MIN_ELEVATION, FEATURE_STYLES


class Investigation:
    def get_x_y_z_from_features(self):
        ID = np.asarray(self.collars["ID"])
        X = np.asarray(self.collars["X"])
        Y = np.asarray(self.collars["Y"])

    def z_coords_on_surface(self, surface):
        ID = np.asarray(self.collars["ID"])
        X = np.asarray(self.collars["X"])
        Y = np.asarray(self.collars["Y"])
        ones = np.ones(len(X))
        Z_max = ones * MAX_ELEVATION
        Z_min = ones * MIN_ELEVATION
        return np.asarray(
            [
                [id, surface.ray_trace([x, y, z_max], [x, y, z_min])[0][0][-1]]
                for id, x, y, z_max, z_min in zip(ID, X, Y, Z_max, Z_min)
            ]
        )

    @property
    def info(self):
        if (
            not self.collars
            and not self.features
            and not self.intervals
            and not self.data
        ):
            print("Empty")
        if hasattr(self, "collars"):
            print("Collars:")
            self.collars.info
        if hasattr(self, "features"):
            print("Features:")
            self.features.info
        if hasattr(self, "intervals"):
            print("Intervals:")
            self.intervals.info
        if hasattr(self, "data"):
            print("Data:")
            self.data.info

    def load_collars(self, collars):
        """ """
        if isinstance(collars, str):
            collars_df = pd.read_csv(collars)
        self.collars = collars_df

    def load_features(self, features):
        """ """
        self.features = pd.read_csv(features) if isinstance(features, str) else features

    def add_features(self, features):
        new_features = pd.read_csv(features) if isinstance(features, str) else features
        if hasattr(self, "features"):
            self.features = pd.concat([self.features, new_features], ignore_index=True)
        else:
            self.features = (
                pd.read_csv(features) if isinstance(features, str) else features
            )

    def add_data(self, data):
        new_data = pd.read_csv(data) if isinstance(data, str) else data
        if hasattr(self, "data"):
            self.data = pd.concat([self.data, new_data], ignore_index=True)
        else:
            self.data = new_data

    def feature_points(self, features="All"):
        """Get the coordinates of the features by merging with the collars DataFrame
        Args:
            features (str or list): String or list of strings with the description of features that should be included.
        """

        if isinstance(features, str) and features != "All":
            features_to_include = self.features.loc[self.features["col1"] == features]
        if isinstance(features, list):
            features_to_include = self.features.loc[
                self.features["col1"].isin(features)
            ]
        else:
            features_to_include = self.features

        self.collars.loc[self.collars["ID"] == features_to_include["ID"]]

        merged_inner = pd.merge(
            left=self.collars, right=features_to_include, left_on="ID", right_on="ID"
        )

        return merged_inner[["X", "Y", "Z_feature"]].astype("float").to_numpy()

    def plotter(self, plotter=None, feature_styles=FEATURE_STYLES):
        if not plotter:
            plotter = pv.Plotter()
        if hasattr(self, "collars"):
            plotter = plot_collars(self, plotter)
        if hasattr(self, "features"):
            plotter = plot_features(self, plotter, feature_styles)
        # if hasattr(self, "data"):
        #     plotter = plot_data(self, plotter, feature_styles)
        return plotter


class Project:
    def __init__(self, id, description=None):
        self.id = id
        self.description = description
        self.geological_model = None
        self.investigations = {}
        self.earthwork_elements = {}
        self.built_elements = {}

    def info(self):
        model_status = "A Geological Model," if self.geological_model else ""
        print(
            f"ID: {self.id} Description: {self.description}, Contains:{model_status} {len(self.investigations)} investigations,{len(self.earthwork_elements)} earthwork elements, and {len(self.built_elements)} built elements."
        )

    def add_investigation(self, investigation_id, investigation):
        self.investigations[investigation_id] = investigation

    def add_geological_model(self, geological_model):
        self.geological_model = geological_model

    # def create_geological_model(self):
    #     # TODO create convenience function?
    #     geological_model = GeologicalModel

    def save_folder(self, dir_path):
        import os

        os.makedirs(dir_path, exist_ok=True)

        investigations = self.investigations.keys()
        invs_path = dir_path + r"\investigations"
        os.makedirs(invs_path, exist_ok=True)

        for inv in investigations:
            inv_path = invs_path + rf"\{inv}"
            os.makedirs(inv_path, exist_ok=True)

            with open(os.path.join(inv_path, rf"{inv}_collars.csv"), "w") as f:
                f.write(self.investigations[inv].collars.to_csv(line_terminator="\n"))

            with open(os.path.join(inv_path, rf"{inv}_features.csv"), "w") as f:
                f.write(self.investigations[inv].features.to_csv(line_terminator="\n"))

            with open(os.path.join(inv_path, rf"{inv}_data.csv"), "w") as f:
                f.write(self.investigations[inv].data.to_csv(line_terminator="\n"))
