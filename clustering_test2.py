import plotly.plotly as py
import plotly
import pandas as pd
import numpy as np
plotly.tools.set_credentials_file(username='fatemeh.y', api_key='0iEv5OVwhZosMiXUTmv4')
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/alpha_shape.csv')
df.head()


################################################################
import os
import argparse
import read_image_m as RIM
import itk
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax
jjj=[]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data[0], data[1],data[2], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)

parser = argparse.ArgumentParser(
    description = """This program uses ray casting method to detect overhang problem""")
parser.add_argument("-args0", type = str, default = ('D:\\results\\UWO_CASE_1526R_1526R_dissected_overhang_Label.nrrd'), help = "dissected image address")
# parser.add_argument("-args1", type = str, default = ('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\MicroCT\dissected\\2944l_fatemeh2'), help = "dissected image address")
# parser.add_argument("-args1", type = str, default = ('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Kowther\Specimen2501L\Specimen2501L\\2501L_reduced'), help = "dissected image address")
# parser.add_argument("-args1", type = str, default = ('D:\evan_campton\\1601L_FLIPPED_CROPPED_EDITED\\1601L_FLIPPED_CROPPED_EDITED\IMAGES\\1601L_FLIPPED_CROPPED_EDITED'), help = "dissected image address")
args = parser.parse_args()


###########################################################################################
# ext=os.path.splitext(str(( args.args1)))[1]
# m_string3=str(( args.args1))
# if (ext==".nii" or ext==".nrrd" or ext==".nhdr"):
#
#     original_volume=sitk.ReadImage(m_string3)
#     original_array=sitk.GetArrayFromImage(original_volume)
# else:
#     original_volume=RIM.dicom_series_reader(m_string3)
#     original_array=itk.GetArrayFromImage(original_volume)
# # intact_volume=RIM.dicom_series_reader(str(unicode('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L2963L','utf-8')))
########################################################################################
ext=os.path.splitext(str(( args.args0)))[1]
m_string3=str(( args.args0))
if (ext==".nii" or ext==".nrrd" or ext==".nhdr"):

    intact_volume=sitk.ReadImage(m_string3)
    intact_array=sitk.GetArrayFromImage(intact_volume)
else:
    intact_volume=RIM.dicom_series_reader(m_string3)
    intact_array=itk.GetArrayFromImage(intact_volume)
# intact_volume=RIM.dicom_series_reader(str(unicode('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\Data_Sets\Calgary\TBone-2015\TBoneCBCT-2015-10\L2963L','utf-8')))


intact_array_original=intact_array
##########################################################################
data=intact_array
# xmin2,xmax2,ymin2,ymax2,zmin2,zmax2=bbox2_3D(original_array)
xmin,xmax,ymin,ymax,zmin,zmax=bbox2_3D(data)
aa=xmax-xmin
bb=ymax-ymin
cc=zmax-zmin

# aa2=xmax2-xmin2
# bb2=ymax2-ymin2
# cc2=zmax2-zmin2
# scale=(aa2*bb2*cc2)/(aa*bb*cc)
scale=int((intact_array.shape[0]*intact_array.shape[1]*intact_array.shape[2])/(aa*bb*cc))
data[np.where(data==255)]=1

data=np.where(data==1.0)
#
from sklearn.cluster import KMeans

if data[0].__len__()>20:
    # if data[0].__len__()/scale<1:
    #     nn=np.ceil(data[0].__len__()/scale)*2
    # else:
    nn=np.ceil(data[0].__len__()/30)

    if nn<1:
        nn=1


    if scale>400:
        nn=np.ceil(data[0].__len__()/40)
    if scale<=15:
        # nn=300

        nn=np.ceil(data[0].__len__()/scale)
        # nn=np.ceil(data[0].__len__()/30)*2.3
        if nn>300:
            nn=300

    # if scale<=10:
    #     nn=np.ceil(data[0].__len__()/scale)*0.2
    #     if nn>300:
    #         nn=300


    if scale<=8:
        # nn=500/scale
        # nn=np.ceil(data[0].__len__()/30)*3
        nn=np.ceil(math.ceil((data[0].__len__()/math.floor(scale)))/2)
        if nn>350:
            nn=350
    if scale<=2:
        nn=np.ceil(data[0].__len__()/30)*4
        # nn=500
        if nn>500:
            nn=500
        # nn=data[0].__len__()/20
    print("number of clusters")
    print(nn)
    # nn=math.ceil(data[0].__len__())/20
    cluster = KMeans(n_clusters=int(nn),algorithm='full',max_iter=500).fit(np.asarray(data).transpose())
    c2=0
    for i in(np.unique(cluster.labels_)):
        jjj.append((np.where(cluster.labels_==i))[0].size)
        print(jjj)
    print("number of clusters")
    print(nn)
    print("max")
    print(np.asarray(jjj).max())
    for i in range(jjj.__len__()):
        k=jjj.pop()
        if k>48:
            print(k)
            print("overhang has happend in label")
            print(i)
            c2=c2+1

    cluster = KMeans(n_clusters=int(nn),algorithm='full',max_iter=500).fit(np.asarray(data).transpose())
    c1=0
    for i in(np.unique(cluster.labels_)):
        jjj.append((np.where(cluster.labels_==i))[0].size)
        print(jjj)

    print("number of clusters")
    print(nn)
    print("max")
    print(np.asarray(jjj).max())
    for i in range(jjj.__len__()):
        k=jjj.pop()
        if k>48:
            print(k)
            print("overhang has happend in label")
            print(i)
            c1=c1+1

    if c1>0 and c2>0:
        print("overhang has happend")
    else:
        print("No Overhang")

else: print("over hang has not happend")
print("number of points")
print(data[0].__len__())
print('scale')
print(scale)
