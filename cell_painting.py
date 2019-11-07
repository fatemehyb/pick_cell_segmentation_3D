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
import wx
from openmesh import *
from skimage.measure import marching_cubes_lewiner
from scipy.spatial import ConvexHull
X_2=[]
Y_2=[]
Z_2=[]

def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

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
index_changed2=[]
faces2=[]
verts2=[]
def main_normal(myvolume,spacing,verts2,faces2):
        # verts, faces = skimage.measure.marching_cubes(volume, level, spacing=(1,1,1))
        verts, faces, normals, values = marching_cubes_lewiner(myvolume, 400.0, spacing)

        mesh=mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts],
                             faces)
        mesh.mlab_source.dataset.cell_data.scalars = np.zeros(faces.size)
        mesh.actor.mapper.scalar_visibility=True
        mlab.gcf().scene.parallel_projection = True
        # cell_data = mesh.mlab_source.dataset.cell_data
        # cell_data = mesh.mlab_source.dataset
        # s = mlab.pipeline.triangular_mesh_source(mesh.mlab_source.points[:,0],mesh.mlab_source.points[:,1],mesh.mlab_source.points[:,2],mesh.mlab_source.triangles)
        # s.data.cell_data.scalars =np.ones(mesh.mlab_source.triangles.size) # Your data here.
        # surf = mlab.pipeline.surface(s)
        # surf.contour.filled_contours = True
        mesh.mlab_source.update()
        # mlab.show()
        mesh_external=mesh

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



        ################################################################################
        # Some logic to select 'mesh' and the data index when picking.
        def picker_callback2(picker_obj):

            picked = picker_obj.actors




            if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:



                point_id=index_changed2.pop()



                index_to_change2=np.where(point_id==(faces2[0].transpose())[:])
                ##################################################################################mayavi puck surface point python no depth
                for i in range(0,index_to_change2[1].size):
                     mesh.mlab_source.dataset.cell_data.scalars[int(index_to_change2[1][i])]=0
                mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'


                mesh2= mlab.pipeline.set_active_attribute(mesh,cell_scalars='Cell data')
                mlab.pipeline.surface(mesh2)

                ###################################################################################







        def picker_callback(picker_obj):
            # picker_obj.tolerance=1
            picked = picker_obj.actors
            # picker_obj.GetActore()



            if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
                # m.mlab_source.points is the points array underlying the vtk
                # dataset. GetPointId return the index in this array.
                # x_, y_ = np.lib.index_tricks.unravel_index(picker_obj.point_id, s.shape)

                x_2, y_2, z_2 = picker_obj.pick_position
                # mesh.mlab_source.points(picker_obj.point_id)
                # xxx=np.asarray(np.where(verts2==np.asarray(picker_obj.pick_position,dtype=int)),dtype=int)
                # yyy=np.where(faces2==xxx.transpose(1,0))
                # xxx=np.where(verts2[0][0][i]==np.asarray(picker_obj.pick_position[0],dtype=int) and verts2[0][1][i]==np.asarray(picker_obj.pick_position[1],dtype=int) and verts2[0][2][i]==np.asarray(picker_obj.pick_position[2],dtype=int))
                # np.where(picker_obj.pick_position[0]==(verts2[0].transpose())[:][0] and picker_obj.pick_position[1]==(verts2[0].transpose())[:][1] and picker_obj.pick_position[2]==(verts2[0].transpose())[:][2])
                # a=np.zeros(faces2[0].shape[0],dtype=bool)
                # b=np.zeros(faces2[0].shape[0],dtype=bool)
                # c=np.zeros(faces2[0].shape[0],dtype=bool)
                # a[np.where(picker_obj.point_id==(faces2[0].transpose())[:][0])]=True
                # b[np.where(picker_obj.point_id==(faces2[0].transpose())[:][1])]=True
                # c[np.where(picker_obj.point_id==(faces2[0].transpose())[:][2])]=True
                # index_to_change=np.where(np.logical_or(a,np.logical_or(b,c))==True)

                index_to_change2=np.where(picker_obj.point_id==(faces2[0].transpose())[:])
                ##################################################################################mayavi puck surface point python no depth
                for i in range(0,index_to_change2[1].size):
                     mesh.mlab_source.dataset.cell_data.scalars[int(index_to_change2[1][i])]=255
                mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
                # mesh.module_manager.scalar_lut_manager=1
                # mesh.mlab_source.update()
                # mesh.mlab_source.

                mesh2= mlab.pipeline.set_active_attribute(mesh,cell_scalars='Cell data')
                mlab.pipeline.surface(mesh2)

                # mesh.mlab_source.update()
                # wx.Yield()
                ###################################################################################
                index_changed2.append( picker_obj.point_id)





                # x_2, y_2, z_2 = picker_obj.mapper_position
                X_2.append(x_2/spacing[0])
                Y_2.append(y_2/spacing[1])
                Z_2.append(z_2/spacing[2])


                print("Data indices: %f, %f, %f" % (x_2, y_2, z_2))
                print("point ID: %f"% (picker_obj.point_id))
                index_changed.append((picker_obj.pick_position))
                print("cell ID: %f"% (picker_obj.cell_id))
                # index_changed.append(int(picker_obj.cell_id))

        picker_obj=fig.on_mouse_pick(picker_callback,type='cell')
        fig.on_mouse_pick(picker_callback2,type='cell',button='Right')
        # picker_obj.tolerance=0.0005



        mlab.show()

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


        return ((The_Normal2),p1,p2,p3,index_changed)

        # From:
        # http://scikit-image.org/docs/dev/api/skimage.measure.html?highlight=marching_cubes#skimage.measure.marching_cubes
        # http://scikit-image.org/docs/dev/auto_examples/plot_marching_cubes.html
# time.time()
startTime = time.time()

# bytes(path, "utf-8").decode("unicode_escape")

parser = argparse.ArgumentParser(
    description = """This program uses ray casting method to detect overhang problem""")
parser.add_argument("-args0", type = str, default = (('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\dissected_27_feb_2019')), help = "dissected image address")
parser.add_argument("-args1",type=str, default=('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\Specimen2501L\\2501L_reduced'),help="intact image address")
#
# parser.add_argument("-args2", type = int, default = 1000,
#     help = "low")
# parser.add_argument("-args3", type = int, default = 4000,
#     help = "high")
# Get your arguments
args = parser.parse_args()
low=1000
high=4000



######################################################################################################################

#####################################################################
def finish():
    print('\n', "bye", '\n')
    input('Press Enter to quit: ')
###############################################################################################################################



##########################################################################################################################


# read the original volume
ext=os.path.splitext(args.args0)[1]
m_string=args.args0
if (ext==".nii" or ext==".nrrd"):

    input_volume=sitk.ReadImage(m_string)
else:
    input_volume=RIM.dicom_series_reader(m_string)

# spacing=input_volume.GetSpacing()
spacing=[0.12,0.12,0.12]
origin=input_volume.GetOrigin()
try:
    myvolume=sitk.GetArrayFromImage(input_volume)
except:
    myvolume=itk.GetArrayFromImage(input_volume)

# # ######################################################################
# #do binary threshoulding on the intact image
# try:
#     thresholdFilter= sitk.BinaryThresholdImageFilter()
#     my_volume2_thr=thresholdFilter.Execute(input_volume,low,high,255,0)
# except:print(0)
# #
# try:
#     PixelType = itk.ctype('signed short')
#     Dimension = 3
#     ImageType_threshold = itk.Image[PixelType, Dimension]
#     thresholdFilter= itk.BinaryThresholdImageFilter[ImageType_threshold,ImageType_threshold].New()
#     thresholdFilter.SetInput(input_volume)
#
#     thresholdFilter.SetLowerThreshold(low)
#     thresholdFilter.SetUpperThreshold(high)
#     thresholdFilter.SetOutsideValue(0)
#     thresholdFilter.SetInsideValue(255)
#     thresholdFilter.Update()
#     my_volume2_thr=thresholdFilter.GetOutput()
# #
# except:print(0)
# #intact_array=itk.GetArrayFromImage(intact_volume2)
# try:
#     my_array_thr=sitk.GetArrayFromImage(my_volume2_thr)
# except:print(0)
# try:
#     my_array_thr=itk.GetArrayFromImage(my_volume2_thr)
# except:print(0)
# #######################################################################
# ##########################################################################################################################
#
#
# # read the intact volume
# ext=os.path.splitext(args.args1)[1]
# m_string=args.args1
# if (ext==".nii" or ext==".nrrd"):
#
#     input_volume_int=sitk.ReadImage(m_string)
# else:
#     input_volume_int=RIM.dicom_series_reader(m_string)
#
# # spacing=input_volume.GetSpacing()
# spacing=[0.12,0.12,0.12]
# origin=input_volume_int.GetOrigin()
# try:
#     myvolume_int=sitk.GetArrayFromImage(input_volume_int)
# except:
#     myvolume_int=itk.GetArrayFromImage(input_volume_int)
#
# # ######################################################################
# #do binary threshoulding on the intact image
# try:
#     thresholdFilter= sitk.BinaryThresholdImageFilter()
#     my_volume2_int_thr=thresholdFilter.Execute(input_volume_int,low,high,255,0)
# except:print(0)
# #
# try:
#     PixelType = itk.ctype('signed short')
#     Dimension = 3
#     ImageType_threshold = itk.Image[PixelType, Dimension]
#     thresholdFilter= itk.BinaryThresholdImageFilter[ImageType_threshold,ImageType_threshold].New()
#     thresholdFilter.SetInput(input_volume_int)
#
#     thresholdFilter.SetLowerThreshold(low)
#     thresholdFilter.SetUpperThreshold(high)
#     thresholdFilter.SetOutsideValue(0)
#     thresholdFilter.SetInsideValue(255)
#     thresholdFilter.Update()
#     my_volume2_int_thr=thresholdFilter.GetOutput()
# #
# except:print(0)
# #intact_array=itk.GetArrayFromImage(intact_volume2)
# try:
#     my_array_int_thr=sitk.GetArrayFromImage(my_volume2_int_thr)
# except:print(0)
# try:
#     my_array_int_thr=itk.GetArrayFromImage(my_volume2_int_thr)
# except:print(0)
# #######################################################################
# d=my_array_int_thr-my_array_thr
# d[np.where((d == [255.0]))] = [1.0]
# sub_res=np.logical_and(my_array_int_thr,d)
# rmin3, rmax3, cmin3, cmax3, zmin3, zmax3=bbox2_3D(sub_res)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

w1 = myvolume.shape[2]
h1 = myvolume.shape[1]
d1 = myvolume.shape[0]
The_Normal,p1,p2,p3,index_changed=main_normal(myvolume[117:173,458:505,381:390],spacing,verts2,faces2)
hull=ConvexHull(index_changed)

del(myvolume)
#############################################################
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from scipy.spatial import ConvexHull
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# points=index_changed
# edges= list(zip(*points))
# for i in hull.simplices:
#     plt.plot(points[i[0:3][0]], points[i[0:3][1]], points[i[0:3][2]], 'r-')
# ax.plot(edges[0],edges[1],edges[2],'bo')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_xlim3d(-5,5)
# ax.set_ylim3d(-5,5)
# ax.set_zlim3d(-5,5)
# plt.show()
