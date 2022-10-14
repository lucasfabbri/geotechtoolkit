from stringprep import in_table_c11
import pyvista as pv
import numpy as np
import gstools as gs

import random

n_layers = 3

sample_topo = pv.read("../examples/demo/sample_topo.ply")
sample_bedrock = pv.read(r"../examples/demo/sample_bedrock.ply")


maximum = sample_topo.points[:, 2].max()
minimum = sample_bedrock.points[:, 2].min() + 5

x = np.arange(sample_bedrock.bounds[0], sample_bedrock.bounds[1], 1)
y = np.arange(sample_bedrock.bounds[2], sample_bedrock.bounds[3], 1)

boundaries = np.linspace(minimum, maximum, 3)

surfs = []


"""REMOVE THIS"""


def clip(item, surface, above=True, plot=False):

    # if type(surface)
    clipper = pv.Plotter()
    clipper.add_mesh(item, color="r", opacity=0.5)
    clipper.add_mesh(surface, opacity=0.5)

    clipped = item.clip_surface(surface, invert=above)
    try:
        clipper.add_mesh(clipped, color="g")
    except:
        print("Clipping Fail")
    if plot == True:
        clipper.show()

    return clipped


for i, boundary in enumerate(boundaries):
    model = gs.Gaussian(dim=2, var=4, len_scale=10, angles=np.pi / 8.0)
    srf = gs.SRF(model)
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
    surf.save(
        r"C:\Users\Lucas\Desktop\geotechtoolkit\geotechtoolkit\test" + str(i) + ".ply"
    )

trimmed_surfs = []

for i, surf in enumerate(surfs):
    rock_trimmed = surf.clip_surface(sample_bedrock, invert=False)
    topo_trimmed = rock_trimmed.clip_surface(sample_topo, invert=True)
    if i == 0:
        trimmed_surfs.append(topo_trimmed)
    if i == len(surfs) - 1:
        trimmed_surfs.append(topo_trimmed)
    else:
        next_layer_trimmed = clip(topo_trimmed, surf[i + 1], plot=True)
        trimmed_surfs.append(next_layer_trimmed)

p = pv.Plotter()
pv.set_plot_theme("document")
p.add_mesh(sample_bedrock, color="r", opacity=0.5)
p.add_mesh(sample_topo, color="g", opacity=0.5)
for (
    color,
    mesh,
) in zip(["b", "y", "w"], trimmed_surfs):
    p.add_mesh(mesh, color=color, opacity=0.5)

p.export_vtkjs("testrandlayers")
p.show()
