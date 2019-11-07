# mesh created with
# verts, faces = skimage.measure.marching_cubes(volume, level, spacing=(1.0, 1.0, 1.0))

from mayavi import mlab
import vtk
import os
import openmesh as om
import argparse
import itk
import SimpleITK as sitk
import numpy as np
import read_image_m as RIM
import glm
import time
from openmesh import *
from skimage.measure import marching_cubes_lewiner
X_2=[]
Y_2=[]
Z_2=[]

def Cross(a, b):
    x = a.y * b.z - a.z * b.y
    y = a.z * b.x - a.x * b.z
    z = a.x * b.y - a.y * b.x

    # //    return (a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
    return glm.vec3(x, y, z)
def Dot(a, b):
    x = a.x * b.x
    y = a.y * b.y
    z = a.z * b.z

    # //    return (a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
    return glm.vec3(x, y, z)


index_changed=[]
faces2=[]
verts2=[]
def main_normal(mesh,verts,faces):
        # verts, faces = skimage.measure.marching_cubes(volume, level, spacing=(1,1,1))
        # verts, faces, normals, values = marching_cubes_lewiner(myvolume, 400.0, spacing)
        #
        mesh=mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts],
                             faces)
        mesh.mlab_source.dataset.cell_data.scalars = np.zeros(faces.__len__())
        # cell_data = mesh.mlab_source.dataset.cell_data
        # cell_data = mesh.mlab_source.dataset
        # s = mlab.pipeline.triangular_mesh_source(mesh.mlab_source.points[:,0],mesh.mlab_source.points[:,1],mesh.mlab_source.points[:,2],mesh.mlab_source.triangles)
        # s.data.cell_data.scalars =np.ones(mesh.mlab_source.triangles.size) # Your data here.
        # surf = mlab.pipeline.surface(s)
        # surf.contour.filled_contours = True
        mesh.mlab_source.update()
        # mlab.show()

        ########################these two lines give you information about celles try to color cells tomorrow#######################################


        # result=read_trimesh(mesh,myvolume)
        faces2.append(faces)
        verts2.append(verts)


        # A first plot in 3D
        fig = mlab.figure(1)
        # for f in faces:
        #     if faces[f,0]==vertex
        # face_index=verts.index(vertex)

        cursor3d = mlab.points3d(0., 0., 0., mode='axes',
                                        color=(0, 0, 0),
                                        scale_factor=0.5)
        mlab.title('Click on the volume to determine 3 points(consider right hand rule)')
        # mlab.show()



        ################################################################################
        # Some logic to select 'mesh' and the data index when picking.

        def picker_callback(picker_obj):
            picked = picker_obj.actors
            # picker_obj.GetActore()



            # if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
                # m.mlab_source.points is the points array underlying the vtk
                # dataset. GetPointId return the index in this array.
                # x_, y_ = np.lib.index_tricks.unravel_index(picker_obj.point_id, s.shape)

            x_2, y_2, z_2 = picker_obj.pick_position
            # xxx=np.asarray(np.where(verts2==np.asarray(picker_obj.pick_position,dtype=int)),dtype=int)
            # yyy=np.where(faces2==xxx.transpose(1,0))
            # xxx=np.where(verts2[0][0][i]==np.asarray(picker_obj.pick_position[0],dtype=int) and verts2[0][1][i]==np.asarray(picker_obj.pick_position[1],dtype=int) and verts2[0][2][i]==np.asarray(picker_obj.pick_position[2],dtype=int))
            # np.where(picker_obj.pick_position[0]==(verts2[0].transpose())[:][0] and picker_obj.pick_position[1]==(verts2[0].transpose())[:][1] and picker_obj.pick_position[2]==(verts2[0].transpose())[:][2])
            a=np.zeros(verts2[0].size,dtype=bool)
            b=np.zeros(verts2[0].size,dtype=bool)
            c=np.zeros(verts2[0].size,dtype=bool)
            a[np.where(picker_obj.pick_position[0]==(verts2[0].transpose())[:][0])]=True
            b[np.where(picker_obj.pick_position[1]==(verts2[0].transpose())[:][1])]=True
            c[np.where(picker_obj.pick_position[2]==(verts2[0].transpose())[:][2])]=True
            index_to_change=np.where(np.logical_and(a,np.logical_and(b,c))==True)
    #         anim(index_to_change[0][:][0],mesh)
    # # fig.on_mouse_pick(picker_callback)
    #
    #         mlab.show()
            # print(index_to_change)
            # index_to_change=np.where(picker_obj.pick_position[0]==(verts2[0].transpose())[:][0]) and np.where(picker_obj.pick_position[1]==(verts2[0].transpose())[:][1]) and np.where(picker_obj.pick_position[2]==(verts2[0].transpose())[:][2])
            # s.data.cell_data.scalars[int(index_to_change[0][:][0])]=255
            # s.data.cell_data.scalars.name = 'Cell data'
            # s.mlab_source.update()
            #
            # mesh2 = mlab.pipeline.set_active_attribute(s,
            #         cell_scalars=s.data.cell_data.scalars.name)
            # mlab.pipeline.surface(mesh2)
            # anim(index_to_change)
            # mlab.show()
            # mesh.mlab_source.update()
            # print(index_to_change)
            # for i in range(0,verts2[0].shape[0]):
            #
            #     if picker_obj.pick_position[0]==verts2[0][i][0] and picker_obj.pick_position[1]==verts2[0][i][1] and picker_obj.pick_position[2]==verts2[0][i][2]:
            #         print("the point found is")
            #         # a_index=verts2[0][i][:]
            #         # print(verts2[0][i][:])
            #
            #         s.data.cell_data.scalars[i]=0
            #         print(i)
            #         flag=1





            # x_2, y_2, z_2 = picker_obj.mapper_position
            X_2.append(x_2)
            Y_2.append(y_2)
            Z_2.append(z_2)


            print("Data indices: %f, %f, %f" % (x_2, y_2, z_2))
            index_changed.append(index_to_change[0][:][0])
            # return(index_to_change[0][:][0])
            # print("Data indices: %f, %f, %f" % (x_2, y_2, z_2))

        @mlab.animate
        def anim():
            mesh.mlab_source.dataset.cell_data.scalars = np.zeros(faces2[0].shape[0])
            # index_changed=[1000,20000,40321,54387]

            for i in range(0,index_changed.__len__()):

                j=index_changed.pop()


                mesh.mlab_source.dataset.cell_data.scalars[int(j)]=255
                mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
                mesh.mlab_source.update()

                mesh2 = mlab.pipeline.set_active_attribute(mesh,cell_scalars='Cell data')
                mlab.pipeline.surface(mesh2)
                yield

        fig.on_mouse_pick(picker_callback)



        mlab.show()
        anim()
        # # fig.on_mouse_pick(picker_callback)
        #
        mlab.show()
        ######################################################################
        # mesh.mlab_source.dataset.cell_data.scalars = np.ones(f.size)
        # i=0
        # @mlab.animate
        # def anim(i):
        #     # for i in range(0,f.size):
        #         while(1):
        #             if (i>=f.size):
        #                 i=0
        #             else:
        #
        #
        #                 mesh.mlab_source.dataset.cell_data.scalars = np.ones(f.size)
        #                 mesh.mlab_source.dataset.cell_data.scalars[i]=255
        #                 mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
        #                 mesh.mlab_source.update()
        #
        #                 mesh2 = mlab.pipeline.set_active_attribute(mesh,
        #                         cell_scalars='Cell data')
        #                 mlab.pipeline.surface(mesh2)
        #                 i=i+1
        #
        #                 # mlab.show()
        #                 yield
        #
        # anim(i)
        # mlab.show()
        ############################################################################
        p1 = glm.vec3(X_2[0],Y_2[0],Z_2[0])
        p2 = glm.vec3(X_2[1],Y_2[1],Z_2[1])
        p3 = glm.vec3(X_2[2],Y_2[2],Z_2[2])

        # These two vectors are in the plane
        v1 = glm.vec3(p3.x - p1.x,p3.y-p1.y,p3.z-p1.z)
        v2 = glm.vec3(p2.x - p1.x,p2.y-p1.y,p2.z-p1.z)

        # the cross product is a vector normal to the plane
        The_Normal =glm.normalize(Cross(v1, v2))
        The_Normal2=The_Normal
        The_Normal2.x=The_Normal.y
        The_Normal2.y=(The_Normal.x)
        The_Normal2.z=(The_Normal.z)
        # The_Normal3=glm.triangleNormal(p1,p2,p3)
        print("Normal is: ")
        print((The_Normal2))
        # print(The_Normal3)


        return ((The_Normal2),p1,p2,p3)

        # From:
        # http://scikit-image.org/docs/dev/api/skimage.measure.html?highlight=marching_cubes#skimage.measure.marching_cubes
        # http://scikit-image.org/docs/dev/auto_examples/plot_marching_cubes.html




######################################################################################################################

#####################################################################
def finish():
    print('\n', "bye", '\n')
    input('Press Enter to quit: ')
###############################################################################################################################



##########################################################################################################################


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
verts=np.zeros((3,x.__len__()))
# These are the scalar values for each triangle
f = np.mean(t[np.array(triangles)], axis=1)

# Plot it
mesh = mlab.triangular_mesh(x, y, z, triangles,
                            representation='wireframe',
                            opacity=0)
# faces2=triangles
verts[0]=x
verts[1]=y
verts[2]=z
verts=verts.transpose()
triangles=np.asarray(triangles)
The_Normal,p1,p2,p3=main_normal(mesh,verts,triangles)

#############################################################
