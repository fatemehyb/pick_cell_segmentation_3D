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
parser.add_argument("-args0", type = str, default = ('D:\\results\label_no_overhang_15_april_L2501.nrrd'), help = "dissected image address")
args = parser.parse_args()

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
xmin,xmax,ymin,ymax,zmin,zmax=bbox2_3D(data)
aa=xmax-xmin
bb=ymax-ymin
cc=zmax-zmin
data[np.where(data==255)]=1
count=0
for i in range(xmin,xmax,20):
    print(i)
    for j in range(ymin,ymax,20):
        for k in range(zmin,zmax,20):
            L=data[i:i+20,j:j+20,k:k+20]
            if np.sum(L)>50:
                print(np.sum(L))
                print("overhang happend")
            if np.sum==0:
                    count=count+1
vol=aa*bb*cc/(50*50)
print("count")
print(count)
data=np.where(data==1.0)
vol2=vol/data[0].__len__()
nn=vol2
# nn=vol2/20
ax.scatter(data[0], data[1],data[2], c='b', **plot_kwds)

# plt.show()

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
# import scipy.cluster.hierarchy as shc
#
# plt.figure(figsize=(10, 7))
# plt.title("Customer Dendograms")
# dend = shc.dendrogram(shc.linkage(data, method='ward'))
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# nn=data[0].__len__()/10

# nn=620/20
# nn2=data[0].__len__()/nn
# nn=620/20
# nn=data[0].__len__()/10
# nn=120/data[0].__len__()
if data[0].__len__()<200:
    nn=int(data[0].__len__()*data[1].__len__()*data[2].__len__()/(50*50*50))
else:
    nn=int(data[0].__len__()*data[1].__len__()*data[2].__len__()/(50*50*50*40))
nn=math.ceil(data[0].__len__()/100)
cluster = KMeans(n_clusters=nn,algorithm='full',n_init=40,max_iter=1000).fit(np.asarray(data).transpose())
# print(cluster.labels_)
# cluster.fit_predict(data)
# plt.figure(figsize=(10, 7))
# ax.scatter(data[0], data[1],data[2], c=cluster.labels_, cmap='rainbow')
for i in(np.unique(cluster.labels_)):
    jjj.append((np.where(cluster.labels_==i))[0].size)
    print(jjj)
for i in range(jjj.__len__()):
    k=jjj.pop()
    if k>200:
        print(k)
        print("overhang has happend in label")
        print(i)
# plt.show()
