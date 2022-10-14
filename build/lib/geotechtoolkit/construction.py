import numpy as np
import pyvista as pv


class ShallowFoundations:
    """Class for processing shallow foundations

    Attributes:
        list (list): list of all shallow foundations

    """

    def __init__(self):
        self.list = []
        self.dictionary = {}

    def import_ifc(self, ifc_file, types=None, BOTTOM_NORMAL_THRESHOLD=-0.5):
        import ifcopenshell
        from ifcopenshell import geom
        from .utils import get_length_and_width_rectangle

        """ Import shallow foundations into the Project class

        Args: 
            ifc_file: ifc_file as opened in ifcopenshell or path
            
        """
        if isinstance(ifc_file, str):
            ifc_file = ifcopenshell.open(ifc_file)
        if types is None:
            types = ["IfcFooting"]

        settings = geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        self.list = []
        for ifctype in types:
            elements = ifc_file.by_type(ifctype)
            for element in elements:
                attr_dict = element.__dict__
                globalid = str(attr_dict["GlobalId"])
                self.list.append((ifctype, globalid))

                shape = geom.create_shape(settings, element)
                faces = shape.geometry.faces
                verts = shape.geometry.verts

                pv_verts = np.asarray(
                    [
                        [verts[i], verts[i + 1], verts[i + 2]]
                        for i in range(0, len(verts), 3)
                    ]
                )
                pv_faces = np.concatenate(
                    [
                        [3, faces[i], faces[i + 1], faces[i + 2]]
                        for i in range(0, len(faces), 3)
                    ]
                )

                polydata = pv.PolyData(pv_verts, pv_faces)
                polydata.compute_normals(
                    cell_normals=True, point_normals=False, inplace=True
                )

                ids = np.arange(polydata.n_cells)[
                    polydata["Normals"][:, 2] < BOTTOM_NORMAL_THRESHOLD
                ]
                bottom = polydata.extract_cells(ids)

                (
                    width,
                    length,
                ) = get_length_and_width_rectangle(bottom.points)

                self.dictionary[globalid] = {
                    "name": str(attr_dict["Name"]),
                    "width": width,
                    "length": length,
                    "elevation": bottom.points[0][
                        -1
                    ],  # TODO: think about this, lazy way
                    "polydata": polydata,
                    "bottom": bottom,
                }

    def plotter(self, plotter=None):
        if not plotter:
            plotter = pv.Plotter()
        for _, ifcid in self.list:
            plotter.add_mesh(self.dictionary[ifcid]["polydata"])
        return plotter
