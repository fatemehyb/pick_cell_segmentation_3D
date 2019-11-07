import plotly.plotly as py
import plotly
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
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


j=[]
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

data=np.where(data==255.0)
X=np.asarray(data).transpose()
tree = KDTree(X, leaf_size=10)
tree.kernel_density(X[0:15], h=0.1, kernel='gaussian')
# print(tree.query_radius(X[:1], r=0.3, count_only=True))
print(tree.query_radius(X[0:15], r=0.3))
# ax.scatter(data[0], data[1],data[2], c='b', **plot_kwds)
#
# # plt.show()
#
# import matplotlib.pyplot as plt
# import pandas as pd
#
# import numpy as np
# # import scipy.cluster.hierarchy as shc
# #
# # plt.figure(figsize=(10, 7))
# # plt.title("Customer Dendograms")
# # dend = shc.dendrogram(shc.linkage(data, method='ward'))
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.cluster import KMeans
#
# nn=data[0].__len__()/10
# cluster = KMeans(n_clusters=int(nn),algorithm='full').fit(np.asarray(data).transpose())
# # print(cluster.labels_)
# # cluster.fit_predict(data)
# # plt.figure(figsize=(10, 7))
# # ax.scatter(data[0], data[1],data[2], c=cluster.labels_, cmap='rainbow')
# for i in(np.unique(cluster.labels_)):
#     j.append((np.where(cluster.labels_==i))[0].size)
# for i in range(j.__len__()):
#     k=j.pop()
#     if k>20:
#         print("overhang has happend in label")
#         print(i)
# # plt.show()
