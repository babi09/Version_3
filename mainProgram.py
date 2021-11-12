# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import modelDeployment
import funcs_ha_use
from PIL import Image
from nibabel import FileHolder, Nifti1Image
from io import BytesIO
from skimage import measure
import pyvista as pv
# streamlit interface

st.sidebar.title('Organ Detection and Segmentation')
flag_Liver_Model = 0

# upload file
@st.cache
def loadData(dataAddress):
    img_vol = funcs_ha_use.readVolume4(dataAddress)
    return img_vol

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


#uploaded_nii_file = file_selector()

#st.write('You selected `%s`' % uploaded_nii_file)


uploaded_nii_file = st.sidebar.file_uploader("Select file:", type=['nii'])
# print (uploaded_nii_file)

if uploaded_nii_file is not None:
    rr = uploaded_nii_file.read()
    bb = BytesIO(rr)
    fh = FileHolder(fileobj=bb)
    img = Nifti1Image.from_file_map({'header': fh, 'image': fh})


    #img_vol = Image.open(uploaded_nii_file)
    #content = np.array(img_vol)  # pil to cv
    #print('yes')
    img_vol = loadData(img)
    # plot the data
    # using three column
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # plot the slider
    n_slices1 = img_vol.shape[2]
    slice_i1 = col1.slider('Slice - Axial', 0, n_slices1, int(n_slices1 / 2))

    n_slices2 = img_vol.shape[0]
    slice_i2 = col2.slider('Slice - Coronal', 0, n_slices2, int(n_slices2 / 2))

    n_slices3 = img_vol.shape[1]
    slice_i3 = col3.slider('Slice - Sagittal', 0, n_slices3, int(n_slices3 / 2))

# plot volume
    fig, ax = plt.subplots()
    plt.axis('off')
    def plotImage(img_vol, slice_i):
        selected_slice = img_vol[:, :, slice_i, 1]

        ax.imshow(selected_slice, 'gray', interpolation='none')
        return fig
    fig = plotImage(img_vol, slice_i1)
    #plot = st.pyplot(fig)

    #plot coronal view
    fig1, ax1 = plt.subplots()
    plt.axis('off')
    def plotImageCor(img_vol, slice_i):
        selected_slice2 = img_vol[slice_i, :, :, 1]
        print ('image vol: ')
        print(img_vol.shape)
        print('length:')
        print(len(selected_slice2))
        print('dim 2')
        print(img_vol.shape[2])
        #tr = transforms.Affine2D().rotate_deg(90).translate(len(selected_slice2), 0)

        #ax1.imshow(selected_slice2,'gray', transform=tr + ax1.transData,  interpolation='none')
        rotateIm = list(reversed(list(zip(*selected_slice2))))
        ax1.imshow(rotateIm, 'gray', interpolation='none')
        #print(selected_slice2.shape)
        return fig1

    fig1 = plotImageCor(img_vol, slice_i2)

    # plot sagittal view
    fig2, ax2 = plt.subplots()
    plt.axis('off')

    def plotImageSag(img_vol, slice_i):
        selected_slice3 = img_vol[:, slice_i, :, 1]

        rotateIm = list(reversed(list(zip(*selected_slice3))))
        ax2.imshow(rotateIm, 'gray', interpolation='none')
        # print(selected_slice2.shape)
        return fig2

    fig2 = plotImageSag(img_vol, slice_i3)
# select organ to segment
#     option = st.sidebar.selectbox('Select organ', ('Kidneys', 'Liver', 'Pancreas'))
#     segmentation = st.sidebar.button('Perform Segmentation')
    option = st.sidebar.radio('Select Organ to segment', ['None', 'Kidney', 'Liver', 'Pancreas', 'Psoas Muscles'], index=0)

    if option == 'Liver':
        # load segmentation model
        # perform segmentation
        maskSegment, plotmask = modelDeployment.runDeepSegmentationModel('Liver', img)
        # plot segmentation mask
        fig, ax = funcs_ha_use.plotMask(fig, ax, img, maskSegment, slice_i1, 'AX')
        fig1, ax1 = funcs_ha_use.plotMask(fig1, ax1, img, maskSegment, slice_i2, 'CR')
        fig2, ax2 = funcs_ha_use.plotMask(fig2, ax2, img, maskSegment, slice_i3, 'SG')


    # plot the three view (axial, sagittal and coronal)



# plot volume
    plot = col1.pyplot(fig)
    plot = col2.pyplot(fig1)
    plot = col3.pyplot(fig2)
    
    if st.sidebar.button('3D visualisation'):
        verts, faces, normals, values = measure.marching_cubes_lewiner(plotmask, 0.0)
        
        cloud = pv.PolyData(verts).clean()

        surf = cloud.delaunay_3d(alpha=3)
        shell = surf.extract_geometry().triangulate()
        #decimated = shell.decimate(0.4).extract_surface().clean()
        #decimated.compute_normals(cell_normals=True, point_normals=False, inplace=True)

        #centers = decimated.cell_centers()
       # centers.translate(decimated['Normals'] * 10.0)

        p = pv.Plotter(notebook=False)
        p.add_mesh(shell, color="r")
        p.link_views()
        p.show()


