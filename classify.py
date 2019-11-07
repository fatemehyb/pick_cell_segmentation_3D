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
df=(np.where(intact_array==255.0))
scatter = dict(
    mode = "markers",
    name = "y",
    type = "scatter3d",
    x = df[0], y = df[1], z = df[2],
    marker = dict( size=2, color="rgb(23, 190, 207)" )
)
clusters = dict(
    alphahull = 20,
    name = "y",
    opacity = 0.1,
    type = "mesh3d",
    x = df[0], y = df[1], z = df[2]
)
layout = dict(
    title = '3d point clustering',
    scene = dict(
        xaxis = dict( zeroline=False ),
        yaxis = dict( zeroline=False ),
        zaxis = dict( zeroline=False ),
    )
)
fig = dict( data=[scatter, clusters], layout=layout )
# Use py.iplot() for IPython notebook
py.iplot(fig, filename='3d point clustering')
