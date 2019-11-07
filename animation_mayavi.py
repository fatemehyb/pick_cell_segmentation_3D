import numpy as np
from mayavi import mlab

# Create cone
n = 8
t = np.linspace(-np.pi, np.pi, n)
z = np.exp(1j*t)
x = z.real.copy()
y = z.imag.copy()
z = np.zeros_like(x)
triangles = [(0, i, i+1) for i in range(n)]
x = np.r_[0, x]
y = np.r_[0, y]
z = np.r_[1, z]
t = np.r_[0, t]

# These are the scalar values for each triangle
f = np.mean(t[np.array(triangles)], axis=1)

# Plot it
mesh = mlab.triangular_mesh(x, y, z, triangles,
                            representation='wireframe',
                            opacity=0)
# mesh.mlab_source.dataset.cell_data.scalars = np.ones(f.size)
i=0
@mlab.animate
def anim(i):
    # for i in range(0,f.size):
        while(1):
            if (i>=f.size):
                i=0
            else:


                mesh.mlab_source.dataset.cell_data.scalars = np.ones(f.size)
                mesh.mlab_source.dataset.cell_data.scalars[i]=255
                mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
                mesh.mlab_source.update()

                mesh2 = mlab.pipeline.set_active_attribute(mesh,
                        cell_scalars='Cell data')
                mlab.pipeline.surface(mesh2)
                i=i+1

                # mlab.show()
                yield

anim(i)
mlab.show()
# mesh.mlab_source.dataset.cell_data.scalars = np.ones(f.size)
# mesh.mlab_source.dataset.cell_data.scalars[2]=255
# mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
# mesh.mlab_source.update()
#
# mesh2 = mlab.pipeline.set_active_attribute(mesh,
#         cell_scalars='Cell data')
# mlab.pipeline.surface(mesh2)
#
# mlab.show()
