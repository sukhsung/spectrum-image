import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.ndimage import affine_transform

def remove_outlier( SI, threshold_multiplier=5, remove_nn=True):
    # remove outliers that are larger than threshold_multiplier*std + median of each spectrum
    # remove_nn also remove two nearest neighbor pixels
    # Outliers are replaced by medians
    (ny,nx,ne) = SI.shape
    SI_cleaned = SI.copy()
    fig, ax = plt.subplots(1)
    for i in range(ny):
        for j in range(nx):
            cur_spec = SI_cleaned[i,j,:]
            med = np.median(cur_spec)
            mean = np.mean(cur_spec)
            std = np.std( cur_spec)

            if std > med:
                ind_outliers, = np.where( cur_spec>(med+threshold_multiplier*std) )
                for ind_outlier in ind_outliers:
                    ind_outlier = int( ind_outlier )
                    if remove_nn:
                        cur_spec[ ind_outlier-1:ind_outlier+2] = med
                    else:
                        cur_spec[ ind_outlier] = med
                SI_cleaned[i,j,:] = cur_spec
                plt.plot(cur_spec)
                print('hi')
    
    return SI_cleaned


def get_hyperspy_data(hs_SI):
    params=hs_SI.axes_manager
    print(params)
    ch1=np.round(hs_SI.axes_manager[2j].get_axis_dictionary()['offset'],4)
    disp=np.round(hs_SI.axes_manager[2j].get_axis_dictionary()['scale'],4)
    hs_SI.z=int(hs_SI.axes_manager[2j].get_axis_dictionary()['size'])
    energy= np.round(np.arange(ch1,ch1+hs_SI.z*disp,disp),4)
    pxscale = hs_SI.axes_manager[0].get_axis_dictionary()['scale']
    if len(energy)!= hs_SI.z:
        energy = energy[:-1]
    return(energy, hs_SI.data, pxscale, disp, params)

def shear_y_SI( SI, ADF=None, angle=0 ):
    # angle = shear angle in degree
    a = np.tan( angle*np.pi/180 )
    shear_matrix_SI = [[1, a, 0],[0, 1, 0],[0, 0, 1]]
    SI_shear = affine_transform(SI, shear_matrix_SI, order=1)
    if ADF is not None:
        shear_matrix_ADF = [[1, a],[0, 1]]
        ADF_shear =affine_transform(ADF, shear_matrix_ADF, order=1)
        return SI_shear, ADF_shear
    else:
        return SI_shear


def shear_x_SI( SI, ADF=None, angle=0 ):
    # angle = shear angle in degree
    a = np.tan( angle*np.pi/180 )
    shear_matrix_SI = [[1, 0, 0],[a, 1, 0],[0, 0, 1]]
    SI_shear = affine_transform(SI, shear_matrix_SI, order=1)
    if ADF is not None:
        shear_matrix_ADF = [[1, 0],[a, 1]]
        ADF_shear =affine_transform(ADF, shear_matrix_ADF, order=1)
        return SI_shear, ADF_shear
    else:
        return SI_shear