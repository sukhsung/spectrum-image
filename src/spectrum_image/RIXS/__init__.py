import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, RectangleSelector, CheckButtons
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, tqdm_notebook


# import spectrum_image.EELS.EELS_util
# import spectrum_image.EELS.EELS_lineshapes as EELS_lineshapes
# from spectrum_image.EELS.EELS_SI import SpectrumImage
# from spectrum_image.EELS.EELS_LP import LineProfile
# import spectrum_image.EELS.EELS_bgsub as bg
# import spectrum_image.EELS.EELS_edge as EELS_edge

from spectrum_image.RIXS.RIXS_EM import EnergyMap