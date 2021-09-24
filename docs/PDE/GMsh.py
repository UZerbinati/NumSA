#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pygmsh,pyvista
with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(
        [
            [0.0, 0.0],
            [1.0, -0.0],
            [1.0, 1.1],
            [0.0, 1.0],
        ],
        mesh_size=0.5,
    )
    mesh = geom.generate_mesh(algorithm=5)
    mesh.write("test.vtk")
# In[10]:

print(mesh.points)
print(mesh.cells[1][1])

vista = pyvista.read('test.vtk')
vista.plot(show_edges=True)

