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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



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
parser.add_argument("-args0", type = str, default = ('\\\\samba.cs.ucalgary.ca\\fatemeh.yazdanbakhsh\Documents\medical_imaging\Results\label_3d_9_3_oct_2_test.nrrd'), help = "dissected image address")
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

data=np.where(data==1.0)
ax.scatter(data[0], data[1],data[2], c='b', **plot_kwds)
# frame = plt.gca()
# frame.axes.get_xaxis().set_visible(False)
# frame.axes.get_yaxis().set_visible(False)
# df=(np.where(intact_array==255.0))
# scatter = dict(
#     mode = "markers",
#     name = "y",
#     type = "scatter3d",
#     x = df[0], y = df[1], z = df[2],
#     marker = dict( size=2, color="rgb(23, 190, 207)" )
# )
# clusters = dict(
#     alphahull = 20,
#     name = "y",
#     opacity = 0.1,
#     type = "mesh3d",
#     x = df[0], y = df[1], z = df[2]
# )
# layout = dict(
#     title = '3d point clustering',
#     scene = dict(
#         xaxis = dict( zeroline=False ),
#         yaxis = dict( zeroline=False ),
#         zaxis = dict( zeroline=False ),
#     )
# )
# fig = dict( data=[scatter, clusters], layout=layout )
# # Use py.iplot() for IPython notebook
# py.iplot(fig, filename='3d point clustering')
# plot_clusters(data, cluster.KMeans, (), {'n_clusters':6})
plt.show()
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
X=data
y=data
k_means.fit(X)
k_means_predicted = k_means.predict(X)
accuracy = round((np.mean(k_means_predicted==y))*100)
print('Accuracy:'+str(accuracy))
centroids = k_means.cluster_centers_

plt.figure('K-Means on Iris Dataset', figsize=(7,7))
ax = plt.axes(projection = '3d')
ax.scatter(X[0],X[1],X[2], c=y , cmap='Set2', s=50)

# color missclassified data

ax.scatter(X[k_means_predicted!=y,3],X[k_means_predicted!=y,0],X[k_means_predicted!=y,2] ,c='b', s=50)

# plot centroids

ax.scatter(centroids[0,3],centroids[0,0],centroids[0,2] ,c='r', s=50, label='centroid')
ax.scatter(centroids[1,3],centroids[1,0],centroids[1,2] ,c='r', s=50)
ax.scatter(centroids[2,3],centroids[2,0],centroids[2,2] ,c='r', s=50)

ax.legend()
