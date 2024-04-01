import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, RectangleSelector, SpanSelector, CheckButtons, RadioButtons, Button
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, tqdm_notebook
import spectrum_image.SI_lineshapes as ls

import numpy.linalg as LA

class eels_edge:
    def __init__( self, label="", e_bsub=None, e_int=None ):
        self.label  = label
        self.e_bsub = e_bsub
        self.e_int  = e_int

    def from_KEM( self, edge_KEM ):
        self.label = edge_KEM[0]
        self.e_bsub = (edge_KEM[1], edge_KEM[2])
        self.e_int = (edge_KEM[3], edge_KEM[4])

    def __str__( self ):
        s = "{}:".format(self.label)
        if self.e_bsub is not None:
            s+= ", e_bsub ({},{})".format(*self.e_bsub)
        if self.e_int is not None:
            s+= ", e_int ({},{})".format(*self.e_int)
        return s

    def __repr__( self ):
        return self.__str__()


class SIbrowser:
    
    def __init__( self, si, energy, im_adf=None, cmap='gray', figsize=(9,4), **kwargs):
        ######### Initialize browser object ##########
        self.si = si
        self.energy = energy
        self.im_adf = im_adf
        self.im_inel =np.mean(si,axis=(-1))

        self.spectrum1=np.mean(si,axis=(0,1))
        self.spectrum2=np.mean(si,axis=(0,1))
        
        ##############Initialize Display#################
        self.fig=plt.figure(figsize=figsize)

        self.ax= {'inel':None,'spec':None,'bttn':None}
        self.ax['inel']=self.fig.add_axes([0.05,0.3,0.425,0.6]) # Image
        self.ax['spec']=self.fig.add_axes([0.55,0.3,0.425,0.6]) # Spectrum
        self.ax['bttn']=self.fig.add_axes([0.88,0.012,0.1,0.2]) # Button
        self.ax['bttn'].axis('off')
        ##############################################
        # Initialize plot handles
        self.h = {'inel':None, 'spec1':None, 'spec2':None}

        ## Inelastic Image
        self.h['inel']  = self.ax['inel'].matshow(self.im_inel,cmap = cmap)
        self.ax['inel'].set_axis_off()
        self.ax['inel'].set_title('Inelastic image')
        ## Spectra
        self.h['spec1'], =self.ax['spec'].plot(self.energy,self.spectrum1,color='maroon')
        self.h['spec2'], =self.ax['spec'].plot(self.energy,self.spectrum2,color='k',alpha=0)

        self.ax['spec'].set_ylim([self.spectrum1.min(),self.spectrum1.max()])
        self.ax['spec'].set_xlim([self.energy.min(),self.energy.max()])
        self.ax['spec'].set_yticks([])
        self.ax['spec'].set_xlabel('Energy (keV)')
        self.ax['spec'].set_ylabel('Intensity')
        self.ax['spec'].set_title('EELS spectrum')
        ##############################################
        results_dict={}
        for key in ['spectrum','image','roi','energy_span']:
            results_dict[key]=[]
        
        ################### Selectors ###################
        self.ui = {'roi1':None, 'roi2':None, 'spec':None, 'logscale':None}
        self.ui['roi1'] = RectangleSelector(self.ax['inel'], self.onselect_function_real_space, button=[1],
                                        useblit=True ,minspanx=1, minspany=1,spancoords='pixels',
                                        interactive=True,props=dict(facecolor='crimson',edgecolor='crimson',alpha=0.2,fill=True),
                                        handle_props=dict(markersize=2,markerfacecolor='white'))#,ignore_event_outside=True
        
        self.ui['roi2'] = RectangleSelector(self.ax['inel'], self.onselect_function_real_space, button=[3],
                                        useblit=True ,minspanx=1, minspany=1,spancoords='pixels',
                                        interactive=True,props=dict(facecolor='royalblue',edgecolor='royalblue',alpha=0.2,fill=True),
                                        handle_props=dict(markersize=2,markerfacecolor='white'))#,ignore_event_outside=True)
            
        self.ui['span_spec'] = SpanSelector(self.ax['spec'], self.onselect_function_spectrum_space, button=[1],
                                            useblit=True, minspan=1,direction="horizontal",
                                            interactive=True,props=dict(facecolor='green',edgecolor='green',alpha=0.2,fill=True),
                                            grab_range=10, drag_from_anywhere=True)
        

        self.ui['logscale']=CheckButtons(self.ax['bttn'],["Log Scale"],useblit=True ,)
        self.ui['logscale'].on_clicked(self.scale_button)
        

        ############### Event Handlers ###################
    def onselect_function_real_space(self, eclick, erelease):
        
        real_roi1 = np.array(self.ui['roi1'].extents).astype('int')
        real_roi2 = np.array(self.ui['roi2'].extents).astype('int')
        
        self.spectrum1=np.mean(self.si[int(real_roi1[2]):int(real_roi1[3]),int(real_roi1[0]):int(real_roi1[1]),:],axis=(0,1))
        self.spectrum2=np.mean(self.si[int(real_roi2[2]):int(real_roi2[3]),int(real_roi2[0]):int(real_roi2[1]),:],axis=(0,1))

        self.update_spectrum1()
        self.update_spectrum2()

    def onselect_function_spectrum_space(self, xmin, xmax):
        indmin, indmax = np.searchsorted(self.energy, (xmin, xmax))
        en_ranges=[xmin,xmax]
        
        self.update_image([indmin,indmax])


    ################### Update Functions ###################
    def update_spectrum1(self):
        self.h['spec1'].set_ydata(self.spectrum1)
        self.h['spec1'].set_color('maroon')
        self.ax['spec'].set_ylim([min(self.spectrum1.min(),self.spectrum2.min()),max(self.spectrum1.max(),self.spectrum2.max())])
        
    def update_spectrum2(self):
        self.h['spec2'].set_ydata(self.spectrum2)
        self.h['spec2'].set_alpha(1)
        self.h['spec2'].set_color('cadetblue')
        self.ax['spec'].set_ylim([min(self.spectrum1.min(),self.spectrum2.min()),max(self.spectrum1.max(),self.spectrum2.max())])
        
    def update_image(self,  x):
        new_im = np.mean(self.si[:,:,x[0]:x[1]],axis=(-1))
        self.h['inel'].set_array(new_im)
        self.h['inel'].autoscale()
        
    def scale_button(self, event):
        if self.ax['spec'].get_yscale()=='linear':
            
            self.ax['spec'].set_yscale('log')
            self.ax['spec'].set_ylabel('Log Intensity')
            self.ax['spec'].set_yticks([])
        else:
            
            self.ax['spec'].set_yscale('linear')
            self.ax['spec'].set_ylabel('Intensity')
            self.ax['spec'].set_yticks([])




class fitbrowser:

    def __init__( self, si, energy, cmap='gray', figsize=(9,6), lc=False, gfwhm=10, log=True, 
                 ftol=0.0005, gtol=0.00005, xtol=None, maxfev = 50000, method='trf', edge=None):
        
        ## Initialize browser object
        self.si = si
        self.energy = energy
        self.spectrum = np.mean(si,axis=(0,1))
        self.im_inel = np.mean(si,axis=(-1))

        self.bsub = np.zeros_like( self.spectrum )
        self.fit = np.zeros_like( self.spectrum )

        self.fit_check = False
        self.int_check = False
        self.slider_check  = False
        self.slider_window = [0,len(energy)]

        self.lc = lc
        self.gfwhm = gfwhm
        self.log = log
        self.ftol = ftol
        self.gtol = gtol
        self.xtol = xtol
        self.maxfev = maxfev
        self.method = method

        self.bsub_array = np.copy(si)

        self.fitfunction = 'pl'
        self.r = -1
        
        results_dict={}
        for key in ['bsub_spectrum','image','bsub_SI','edge']:
            results_dict[key]=[]

        
        ##############Set Initial plot#################
        self.fig=plt.figure(figsize=figsize,layout='constrained')

        self.ax = {'inel':None, 'spec':None, 'slid':None, 'btn_fit':None, 'btn_fint':None, 'btn_int':None, 'btn_save':None}
        self.ax['inel']=self.fig.add_axes([0.025,0.1,0.45,0.8]) # Image
        self.ax['spec']=self.fig.add_axes([0.525,0.45,0.45,0.45]) # Spectrum
        self.ax['slid']=self.fig.add_axes([0.625,0.3,0.25,0.05]) # Range slider
        self.ax['btn_fit']=self.fig.add_axes([0.520,0.1,0.125,0.1]) # Fit Buttons
        self.ax['btn_fint']=self.fig.add_axes([0.655,0.1,0.1,0.1]) # Fast Int Button
        self.ax['btn_int']=self.fig.add_axes([0.765,0.1,0.1,0.1]) # Int Button 
        self.ax['btn_save']=self.fig.add_axes([0.875,0.1,0.1,0.1]) # Save Button 

        ## Initialize plot handles
        self.h = {'inel':None, 'spec':None, 'bsub':None, 'fit':None}
        ################## ax['inel'] ######################
        self.h['inel'] = self.ax['inel'].imshow( self.im_inel,cmap = cmap)
        self.ax['inel'].set_axis_off()
        self.ax['inel'].set_title('Inelastic image')

        ###################self.ax['spec']########################
        self.h['spec'], =self.ax['spec'].plot(energy, self.spectrum,color='k')
        self.h['bsub'], =self.ax['spec'].plot(energy, self.bsub,color='k',alpha=0)
        self.h['fit'],  =self.ax['spec'].plot(energy, self.fit,color='k',alpha=0)
        self.ax['spec'].axhline(0,color='k',linestyle='--',alpha=0.3)
        self.ax['spec'].set_ylim([self.spectrum.min(),self.spectrum.max()])
        self.ax['spec'].set_xlim([self.energy.min(),self.energy.max()])
        self.ax['spec'].set_yticks([])
        self.ax['spec'].set_xlabel('Energy (keV)')
        self.ax['spec'].set_ylabel('Intensity')
        self.ax['spec'].set_title('EELS spectrum')

        ## Initialize ui handles
        self.ui = {'roi':None, 'spec1':None, 'spec2':None, 'slid_e':None, 'rad_fit':None, 'btn_fint':None, 'btn_int':None, 'btn_save':None}

        ################### Selectors ###################
        self.ui['roi'] = RectangleSelector(self.ax['inel'], self.onselect_function_real_space, button=[1],
                                        useblit=True ,minspanx=1, minspany=1,spancoords='pixels',
                                        interactive=True,props=dict(facecolor='crimson',edgecolor='crimson',alpha=0.2,fill=True),
                                        handle_props=dict(markersize=2,markerfacecolor='white'))#,ignore_event_outside=True
        
            
        self.ui['bsub'] = SpanSelector(self.ax['spec'], self.onselect_function_spectrum_space1, button=[1],
                                            useblit=True, minspan=1,direction="horizontal",
                                            interactive=True,props=dict(facecolor='C0',edgecolor='C0',alpha=0.2,fill=True),
                                            grab_range=10, drag_from_anywhere=True)
        
        self.ui['int'] = SpanSelector(self.ax['spec'], self.onselect_function_spectrum_space2, button=[3],
                                            useblit=True, minspan=1,direction="horizontal",
                                            interactive=True,props=dict(facecolor='orange',edgecolor='orange',alpha=0.2,fill=True),
                                            grab_range=10, drag_from_anywhere=True)
        

        if edge is None:
            self.edge = eels_edge( " ", (energy[0],energy[-1]), (energy[0],energy[-1]) )
        else:
            self.edge= edge

            self.ui['bsub'].extents = self.edge.e_bsub
            self.onselect_function_spectrum_space1()

            self.ui['int'].extents = self.edge.e_int
            self.onselect_function_spectrum_space2()
        
        
        self.ui['slid_e'] = Eslider=RangeSlider(self.ax['slid'],"Energy Range ",energy[0],energy[-1],valinit=[energy[0],energy[-1]],
                        valstep=energy[1]-energy[0],dragging=True)
        self.ui['slid_e'].on_changed( self.slider_action )


        self.ax['btn_fit'].set_facecolor('0.85')
        self.ui['rad_fit'] = RadioButtons(self.ax['btn_fit'], ('Power law', 'Exponential', 'Linear'),
                            label_props={'color': ['k','k','k'], 'fontsize': [10, 10, 10]},
                            radio_props={'s': [16,16,16]})
        
        self.ui['rad_fit'].on_clicked(self.fitcheck)

        self.ui['btn_fint']=Button(self.ax['btn_fint'],"Fast Integrate",useblit=True,)
        self.ui['btn_fint'].on_clicked(self.fint_button)

        self.ui['btn_int']=Button(self.ax['btn_int'],"Integrate",useblit=True,)
        self.ui['btn_int'].on_clicked(self.int_button)

        self.ui['btn_save']=Button(self.ax['btn_save'],'Save results')
        # self.ui['btn_save'].on_clicked(self.add_save_dict)
        
        # selector_collection = (spectrum_span_selector1,spectrum_span_selector2,rect_selector,ibutton,fibutton,Eslider,save_button,fitradio)
        
        
        # return results_dict,selector_collection

    ################### Update Functions ###################
    def update_spectrum(self):
        self.h['spec'].set_ydata( self. spectrum)
        self.h['spec'].set_color('k')
        self.rescale_yrange()
        
    def update_fit(self,  ind):
        self.h['bsub'].set_ydata(self.bsub)
        self.h['bsub'].set_color('C1')
        self.h['bsub'].set_alpha(1)
        self.h['fit'].set_data( self.energy[ind:], self.spectrum[ind:]-self.bsub[ind:])
        self.h['fit'].set_color('C0')
        self.h['fit'].set_alpha(1)
        self.rescale_yrange()
        
    def update_image(self):
        self.bsub_array = bgsub_fast(self.si, self.energy,self.edge.e_bsub,self.r,fit = self.fitfunction)
        indmin, indmax = np.searchsorted(self.energy, self.edge.e_int)
        self.im_inel = np.mean( self.bsub_array[:,:,indmin:indmax],axis=(-1))
        self.h['inel'].set_array( self.im_inel)
        self.h['inel'].autoscale()
        
    def update_image_2(self):

        if self.lc:
            _,self.bsub_array = bgsub_SI(self.si, self.energy,self.edge.e_bsub,gfwhm=self.gfwhm,fit=self.fitfunction,lc=self.lc,log=self.log,
                                    ftol=self.ftol,gtol=self.gtol,xtol=self.xtol,maxfev=self.maxfev,method=self.method)
        else:
            self.bsub_array =   bgsub_SI(self.si, self.energy,self.edge.e_bsub,gfwhm=self.gfwhm,fit=self.fitfunction,lc=self.lc,log=self.log,
                                    ftol=self.ftol,gtol=self.gtol,xtol=self.xtol,maxfev=self.maxfev,method=self.method)
        indmin, indmax = np.searchsorted(self.energy, self.edge.e_int)
        self.im_inel = np.mean(self.bsub_array[:,:,indmin:indmax],axis=(-1))
        self.h['inel'].set_array(self.im_inel)
        self.h['inel'].autoscale()

    def slider_action(self, erange):

        self.slider_check=True
        slidermin, slidermax = np.searchsorted(self.energy, (erange[0],erange[1]))
        self.slider_window = [slidermin, slidermax]
        self.ax['spec'].set_xlim([erange[0],erange[1]])

        self.rescale_yrange()

    def rescale_yrange(self):
        if self.slider_check:
            slidermin,slidermax = self.slider_window
            if self.fit_check:
                self.ax['spec'].set_ylim([min(self.bsub[slidermin:slidermax].min(),-1*self.bsub[slidermin:slidermax].min()),self.spectrum[slidermin:slidermax].max()])
            else:
                self.ax['spec'].set_ylim([self.spectrum[slidermin:slidermax].min(),self.spectrum[slidermin:slidermax].max()])
        else:
            if self.fit_check:
                self.ax['spec'].set_ylim([min(self.bsub.min(),-1*self.bsub.min()),self.spectrum.max()])
            else:
                self.ax['spec'].set_ylim([self.spectrum.min(),self.spectrum.max()])

    ############### Event Handlers ###################
    def onselect_function_real_space(self, eclick, erelease):
        
        real_roi = np.array( self.ui['roi'].extents).astype('int')
        self.spectrum=np.mean( self.si[int(real_roi[2]):int(real_roi[3]),int(real_roi[0]):int(real_roi[1]),:],axis=(0,1))

        self.update_spectrum()
        
        if self.fit_check:
            [xmin,xmax] = self.edge.e_bsub
            indmin, indmax = np.searchsorted(self.energy, (xmin, xmax))
            self.bsub, self.r = bgsub_1D(self.spectrum,self.energy,self.edge.e_bsub,fit =  self.fitfunction)
            self.update_fit(indmin)

    def onselect_function_spectrum_space1( self, e_min, e_max ):
        
        self.fit_check = True
        indmin, indmax = np.searchsorted( self.energy, (e_min, e_max) ) 
        self.edge.e_bsub = ( self.energy[indmin], self.energy[indmax] )


        self.bsub, self.r = bgsub_1D(self.spectrum, self.energy, self.edge.e_bsub, fit = self.fitfunction)

        self.update_fit(indmin)

    def onselect_function_spectrum_space2( self, e_min, e_max ):

        self.int_check = True
        indmin, indmax = np.searchsorted(self.energy, (e_min, e_max) ) 
        self.edge.e_int = (self.energy[indmin],self.energy[indmax])

    def fitcheck(self, label):
        fitdict = {'Power law': 'pl', 'Exponential': 'exp', 'Linear': 'lin'}
        self.fitfunction = fitdict[label]
        if self.fit_check:
            [xmin,xmax] = self.edge.e_bsub
            indmin, indmax = np.searchsorted(self.energy, (xmin, xmax))
            self.bsub, self.r = bgsub_1D(self.spectrum,self.energy,self.edge.e_bsub,fit = self.fitfunction)
            self.update_fit(indmin)
        
    def fint_button(self, event):
        if (self.int_check & self.fit_check):
            self.update_image()

    def int_button(self, event):
        if (self.int_check & self.fit_check):
            self.update_image_2()

    # def add_save_dict(event):
    #     results_dict['bsub_spectrum'] = bsub
    #     results_dict['image'] = inel_im
    #     results_dict['bsub_SI'] = bsub_array
    #     results_dict['edge'] = [fit_window[0],fit_window[1],int_window[0],int_window[1]]




class SI :
    def __init__( self, im_si, ADF=[], es=[] ):

        im_si[np.isnan(im_si)] = 0
        self.si = im_si
        (self.ny, self.nx, self.ne) = self.si.shape
        self.mean_spectrum = np.mean( self.si, axis=(0,1))
        self.set_roi( [self.nx/4, 3*self.nx/4, self.ny/4, 3*self.ny/4] )
        
        ADF = np.asarray( ADF )
        if not ADF.any():
            ADF = np.mean( self.si, axis=2 )

        (ny, nx) = ADF.shape
        if (ny!=self.ny) or (nx!=self.nx):
            print( "ADF size doesn't match SI")
            ADF = np.mean( self.si, axis=2 )
        self.ADF = ADF

        es  = np.asarray( es )
        if not es.any():
            es = np.arange(0, self.ne) +1
        self.es = es

        self.bg_type = None
        self.e_bsub = (self.es[0], self.es[int(self.ne/8)])
        self.e_int  = (self.es[0], self.es[int(self.ne/8)])

    def set_roi( self, roi ):
        xi = int( roi[0] )
        xf = int( roi[1] )
        yi = int( roi[2] )
        yf = int( roi[3] )
        if xf == xi:
            xf = xi+1
        if yf == yi:
            yf = yi+1
        self.roi = (xi, xf, yi, yf)
        self.roi_spectrum = np.mean( self.si[ yi:yf, xi:xf,:], axis=(0,1))
    
    def bin_energy_x2( self ):
        if ( self.ne%2 ==1 ):
            self.ne -= 1
            self.es = self.es[0:-1]
            self.si = self.si[:,:,0:-1]

        self.es = ( self.es[0::2]+self.es[1::2] )/2
        self.si = ( self.si[:,:,0::2] + self.si[:,:,1::2 ] )/2

        (self.ny, self.nx, self.ne) = self.si.shape
        self.mean_spectrum = np.mean( self.si, axis=(0,1))
        self.set_roi( [self.nx/4, 3*self.nx/4, self.ny/4, 3*self.ny/4] )
        self.e_bsub = (self.es[0], self.es[int(self.ne/8)])
        self.e_int  = (self.es[0], self.es[int(self.ne/8)])

    def bin_x_x2( self ):
        if ( self.nx%2 ==1 ):
            self.nx -= 1
            self.si = self.si[:,0:-1,:]
            self.ADF = self.ADF[:,0:-1]

        self.si = ( self.si[:,0::2,:] + self.si[:,1::2,: ] )/2
        self.ADF = ( self.ADF[:,0::2] + self.ADF[:,1::2 ] )/2
        (self.ny, self.nx, self.ne) = self.si.shape
        self.set_roi( [self.nx/4, 3*self.nx/4, self.ny/4, 3*self.ny/4] )

    def bin_y_x2( self ):
        if ( self.ny%2 ==1 ):
            self.ny -= 1
            self.si = self.si[0:-1,:,:]
            self.ADF = self.ADF[0:-1,:]

        self.si = ( self.si[0::2,:,:] + self.si[1::2,:,: ] )/2
        self.ADF = ( self.ADF[0::2,:] + self.ADF[1::2,: ] )/2
        (self.ny, self.nx, self.ne) = self.si.shape
        self.set_roi( [self.nx/4, 3*self.nx/4, self.ny/4, 3*self.ny/4] )

    def bin_xy_x2( self ):
        self.bin_x_x2()
        self.bin_y_x2()


    def SI_viewer( self, bg_type=None):
        self.bg_type = bg_type
        SI_max = np.max(self.si)

        fig = plt.figure(figsize=(6,5))
        self.ax_ADF  =      fig.add_axes([0.1, 0.65, 0.9, 0.3])
        self.ax_spec =      fig.add_axes([0.1,  0.2, 0.8, 0.4])
        self.ax_slider_e =  fig.add_axes([0.2, 0.05, 0.6, 0.03])
        self.ax_slider_bg = fig.add_axes([0.2, 0.01, 0.6, 0.03])
        self.ax_ylock     = fig.add_axes([0.75, 0.1, 0.2, 0.03])

        self.ax_ADF.matshow( self.ADF )
        if bg_type is not None:
            self.rect_bg = patches.Rectangle( (self.e_bsub[0], -SI_max), 
                                               self.e_bsub[1]-self.e_bsub[0], 2*SI_max,
                                               linewidth=1, edgecolor='r', facecolor='red', alpha=0.5)
            self.ax_spec.add_patch( self.rect_bg )
        self.ax_spec.axhline( y=0, color='k')
        self.p_spec,   = self.ax_spec.plot( self.es, self.roi_spectrum )
        self.p_bg,     = self.ax_spec.plot( self.es, np.zeros_like( self.es ) )
        self.p_bgsub,  = self.ax_spec.plot( self.es, np.zeros_like( self.es ) )
        
        self.ax_spec.set_ylim( [-0.1*np.max(self.roi_spectrum), 1.2*np.max(self.roi_spectrum)])
        self.ax_spec.set_xlim( [self.es[0], self.es[-1]])

        props = dict(edgecolor='red', facecolor='none', alpha=1)
        self.rect_roi = RectangleSelector(self.ax_ADF, self.dummy, 
                                        interactive=True,
                                        drag_from_anywhere=True,
                                        props=props)
        self.rect_roi.extents= self.roi

        # Create Checkbox for ylim locker
        self.chkbox_ylock = CheckButtons(
            ax=self.ax_ylock,
            labels= ["Lock Y-axis"],
            actives=[False],
            check_props={'facecolor': 'k'},
        )
        self.ylock = False
        self.chkbox_ylock.on_clicked( lambda value: self.onclick_ylock() )

        # Create the RangeSlider
        if bg_type is not None:
            self.slider_bg = RangeSlider(self.ax_slider_bg, "Background", 
                                         self.es[0], self.es[-1])
            self.slider_bg.set_min( self.e_bsub[0] )
            self.slider_bg.set_max( self.e_bsub[1] )
        else:
            self.slider_bg = None
            self.ax_slider_bg.set_axis_off()


        self.slider_e = RangeSlider(self.ax_slider_e, "Energy Range", self.es[0], self.es[-1])
        self.slider_e.set_min( self.es[0] )
        self.slider_e.set_max( self.es[-1] )


        cid = fig.canvas.mpl_connect('motion_notify_event', 
                                    lambda event: self.onchange_roi(event) )
        if bg_type is not None:
            self.slider_bg.on_changed( lambda value: self.onchange_slider_bg() )
        
        self.slider_e.on_changed( lambda value: self.onchange_slider_e() )


    
    def SI_integrator( self ):
        SI_max = np.max(self.si)

        fig = plt.figure(figsize=(6,5))
        self.ax_ADF  =      fig.add_axes([0.2, 0.65, 0.3, 0.3])
        self.ax_int  =      fig.add_axes([0.6, 0.65, 0.3, 0.3])
        self.ax_spec =      fig.add_axes([0.1,  0.2, 0.9, 0.4])
        self.ax_slider_e   = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        self.ax_slider_int = fig.add_axes([0.2, 0.01, 0.6, 0.03])
        self.ax_ylock     = fig.add_axes([0.75, 0.1, 0.2, 0.03])

        self.ax_ADF.matshow( self.ADF )
        im_int,dummy = self.integrate_SI()
        self.h_int = self.ax_int.matshow( im_int, vmin=0, vmax=1 )
        self.rect_int = patches.Rectangle( (self.e_int[0], -SI_max), 
                                            self.e_int[1]-self.e_int[0], 2*SI_max,
                                            linewidth=1, edgecolor='y', facecolor='yellow', alpha=0.5)
        self.ax_spec.add_patch( self.rect_int )
        self.ax_spec.axhline( y=0, color='k')
        self.p_spec,   = self.ax_spec.plot( self.es, self.roi_spectrum )
        self.p_bg,     = self.ax_spec.plot( self.es, np.zeros_like( self.es ) )
        self.p_bgsub,  = self.ax_spec.plot( self.es, np.zeros_like( self.es ) )
        
        min_y = np.min([-0.1*np.max(self.roi_spectrum), 1.2*np.min(self.roi_spectrum)])
        self.ax_spec.set_ylim( [min_y, 1.2*np.max(self.roi_spectrum)])
        self.ax_spec.set_xlim( [self.es[0], self.es[-1]])

        props = dict(edgecolor='red', facecolor='none', alpha=1)
        self.rect_roi = RectangleSelector(self.ax_ADF, self.dummy, 
                                        interactive=True,
                                        drag_from_anywhere=True,
                                        props=props)
        self.rect_roi.extents= self.roi


        # Create Checkbox for ylim locker
        self.chkbox_ylock = CheckButtons(
            ax=self.ax_ylock,
            labels= ["Lock Y-axis"],
            actives=[False],
            check_props={'facecolor': 'k'},
        )
        self.ylock = False
        self.chkbox_ylock.on_clicked( lambda value: self.onclick_ylock() )

        # Create the RangeSlider
        self.slider_int = RangeSlider(self.ax_slider_int, "Integration", 
                                        self.es[0], self.es[-1])
        self.slider_int.set_min( self.e_int[0] )
        self.slider_int.set_max( self.e_int[1] )


        self.slider_e = RangeSlider(self.ax_slider_e, "Energy Range", self.es[0], self.es[-1])
        self.slider_e.set_min( self.es[0] )
        self.slider_e.set_max( self.es[-1] )


        cid = fig.canvas.mpl_connect('motion_notify_event', 
                                    lambda event: self.onchange_roi(event) )
        
        self.slider_int.on_changed( lambda value: self.onchange_slider_int() )
        self.slider_e.on_changed( lambda value: self.onchange_slider_e() )
        
        self.update_spectrum( )
        self.update_xylim()

    def onclick_ylock( self ):
        self.ylock = self.chkbox_ylock.get_status()[0]
        if self.ylock==False:
            self.update_xylim()

    def onchange_roi( self, event):
        if not (event.button == None):
            self.set_roi( self.rect_roi.extents )
            self.update_spectrum( )
            self.update_xylim()
            if self.bg_type is not None:
                self.update_bg()

    def update_xylim( self ):
        self.ax_spec.set_xlim( self.slider_e.val )
        ei_ch = self.eVtoCh( self.slider_e.val[0], self.es)
        ef_ch = self.eVtoCh( self.slider_e.val[1], self.es)
        if not self.ylock:
            max_spec = np.max(self.roi_spectrum[ei_ch:ef_ch] )
            min_y = np.min([-0.1*max_spec, 1.2*np.min(self.roi_spectrum[ei_ch:ef_ch])])
            self.ax_spec.set_ylim( [min_y, 1.2*max_spec])
        
    def update_spectrum( self ):
            self.p_spec.set_ydata( self.roi_spectrum)

    def onchange_slider_e( self ):
        self.update_xylim()

    def onchange_slider_bg( self ):
        self.e_bsub = self.slider_bg.val
        self.rect_bg.set_x( self.e_bsub[0] )
        self.rect_bg.set_width( self.e_bsub[1]-self.e_bsub[0] )
        self.update_bg( )

    def onchange_slider_int( self ):
        self.e_int = self.slider_int.val
        self.rect_int.set_x( self.e_int[0] )
        self.rect_int.set_width( self.e_int[1]-self.e_int[0] )
        im_int,dummy = self.integrate_SI()
        self.h_int.set_data( self.normalize( im_int ) )

    def update_bg( self ):
        fit_i_ch = self.eVtoCh( self.e_bsub[0], self.es)
        fit_f_ch = self.eVtoCh( self.e_bsub[1], self.es)

        e_fit = self.es[fit_i_ch:fit_f_ch]
        y_fit = self.roi_spectrum[fit_i_ch:fit_f_ch] 

        ftol = 0.0005
        gtol = 0.00005        
                    
        bg_func, d_func, p0, bounds = self.prepare_bgfit( y_fit, e_fit )
        try:
            p_fit, cov_fit = curve_fit( bg_func, e_fit, y_fit, p0=p0, bounds=bounds,
                                    method='trf', jac=d_func, ftol=ftol, gtol=gtol)
            bg_fit = bg_func( self.es, *p_fit)
        except:
            bg_fit = 0*self.roi_spectrum
        
        bsub = self.roi_spectrum-bg_fit
        bsub[0:fit_i_ch] = 0
        self.p_bg.set_ydata( bg_fit )
        self.p_bgsub.set_ydata( bsub )

    def prepare_bgfit( self, y_fit, e_fit ):
        # for Powerlaw and LCPL
        r_g = -np.log( y_fit[0]/y_fit[-1] )/np.log( e_fit[0]/e_fit[-1] )


        if r_g < 0:
            r_g = 0
        elif r_g >5:
            r_g = 5
        A_g = y_fit[0]*(e_fit[0]**r_g)
        # r_g = 3
        # print(r_g)

        if self.bg_type == "LCPL":
            params = 0.5*A_g, r_g, 0.5*A_g, r_g
            bounds = ( [     0,  0,      0,  0], 
                       [np.inf, 10, np.inf, 10] )
            bg_func = ls.LCPL
            d_func = ls.d_LCPL
        elif self.bg_type == "powerlaw":
            params = A_g, r_g
            bounds = ( [     0,  0], 
                       [np.inf, 10] )
            bg_func = ls.powerlaw
            d_func = ls.d_powerlaw
        # d_func = '2-point'
        return bg_func, d_func, params, bounds

    def integrate_SI( self, e_int=None ):
        if e_int is not None:
            self.e_int = e_int
        int_i_ch = self.eVtoCh(self.e_int[0], self.es)
        int_f_ch = self.eVtoCh(self.e_int[1], self.es)

        return np.sum( self.si[:,:, int_i_ch:int_f_ch], axis=(2) ), self.e_int

    def normalize( self, x ):
        if np.min(x) == np.max(x):
            return x
        else:
            return (x-np.min(x))/(np.max(x)-np.min(x))

    def bgsub_SI( self, e_bsub=None, bg_type='powerlaw', LCPL_percentile=(5,95), lba=False, plot_bg_coef= False, lba_gfwhm=5, ftol=0.0005, gtol=0.00005):
        if bg_type=='powerlaw':
            return self.bgsub_powerlaw_SI( e_bsub=e_bsub, lba=lba, plot_bg_coef=plot_bg_coef, lba_gfwhm=lba_gfwhm, ftol=ftol, gtol=gtol)
        elif bg_type=='LCPL_Fixed':
            return self.bgsub_LCPL_Fixed_SI( e_bsub=e_bsub, lba=lba, LCPL_percentile=(5,95), plot_bg_coef=plot_bg_coef, lba_gfwhm=lba_gfwhm, ftol=ftol, gtol=gtol)
        elif bg_type=='LCPL_Free':
            return self.bgsub_LCPL_Free_SI( e_bsub=e_bsub, lba=lba, plot_bg_coef=plot_bg_coef, lba_gfwhm=lba_gfwhm, ftol=ftol, gtol=gtol)

    def bgsub_powerlaw_SI( self, e_bsub=None, lba=False, plot_bg_coef= False, lba_gfwhm=5, ftol=0.0005, gtol=0.0005):
        if e_bsub is not None:
            self.e_bsub = e_bsub

        self.bg_type = 'powerlaw'
        bg_pl_SI = np.zeros_like(self.si)
        fit_i_ch = self.eVtoCh(self.e_bsub[0], self.es)
        fit_f_ch = self.eVtoCh(self.e_bsub[1], self.es)
        e_fit = self.es[fit_i_ch:fit_f_ch]

        #### Setup for LBA if necessary
        if lba :
            si_fit = self.prepare_lba( lba_gfwhm=lba_gfwhm )
        else :
            si_fit = self.si.copy()[:,:,fit_i_ch:fit_f_ch]
    
        ## Fit Mean Spectrum to get a good guess Parameter
        mean_spec = np.mean( si_fit, axis=(0,1))
        y_fit = mean_spec
        r1s = np.zeros( (self.ny,self.nx) )
        A1s = np.zeros( (self.ny,self.nx) )
        bg_func, d_func, p0, bounds = self.prepare_bgfit( y_fit, e_fit )

        ## perform PL fit and background subtraction on each pixel.
        pbar = tqdm_notebook(total = (self.nx)*(self.ny),desc = "Background subtracting")
        for i in range(self.ny):
            for j in range(self.nx):
                y_fit = si_fit[i,j]
                try:
                    p_fit, cov_fit = curve_fit( bg_func, e_fit, y_fit, p0=p0, bounds=bounds,
                                                method='trf',jac=d_func, ftol=ftol, gtol=gtol, maxfev=5000)
                    bg_fit = bg_func( self.es, *p_fit)

                    A1s[i,j] = p_fit[0]
                    r1s[i,j] = p_fit[1]
                except:
                    bg_fit = 0

                bsub = self.si[i,j]-bg_fit
                bsub[0:fit_i_ch] = 0
                bg_pl_SI[i,j,:] = bsub
                pbar.update(1)
        pbar.close()
                
        if plot_bg_coef:
            fig,ax = plt.subplots(1,2)
            fig.suptitle( 'A*E^(-r)' )
            ax[0].matshow( A1s )
            ax[0].set_title( 'A' )
            ax[1].matshow( r1s )
            ax[1].set_title( 'r' )

        return bg_pl_SI, self.e_bsub
    

    def bgsub_LCPL_Free_SI( self, e_bsub=None, lba=False, plot_bg_coef=False, lba_gfwhm=5, ftol=0.0005, gtol=0.00005):#,xtol=None):
        if e_bsub is not None:
            self.e_bsub = e_bsub


        #### Do standard Powerlaw fitting to generate guesses
        self.bg_type = 'powerlaw'
        bg_pl_SI = np.zeros_like(self.si)
        fit_i_ch = self.eVtoCh(self.e_bsub[0], self.es)
        fit_f_ch = self.eVtoCh(self.e_bsub[1], self.es)
        e_fit = self.es[fit_i_ch:fit_f_ch]

        #### Setup for LBA if requested
        if lba :
            si_fit = self.prepare_lba( lba_gfwhm=lba_gfwhm )
        else :
            si_fit = self.si.copy()[:,:,fit_i_ch:fit_f_ch]
    
        ## Fit Mean Spectrum to get a good guess Parameter
        mean_spec = np.mean( si_fit, axis=(0,1))
        y_fit = mean_spec        
        bg_func, d_func, p0, bounds = self.prepare_bgfit( y_fit, e_fit )
        
        ## perform PL fit on each pixel.
        rs = np.zeros( (self.ny, self.nx) )
        As = np.zeros( (self.ny, self.nx) )
        pbar = tqdm_notebook(total = (self.nx)*(self.ny),desc = "Fitting Normal Powerlaw")
        for i in range(self.ny):
            for j in range(self.nx):
                y_fit = si_fit[i,j]
                
                p_fit, cov_fit = curve_fit( bg_func, e_fit, y_fit, p0=p0, bounds=bounds,
                                           method='trf',jac=d_func, ftol=ftol, gtol=gtol, maxfev=5000)
            
                As[i,j] = p_fit[0]
                rs[i,j] = p_fit[1]
                pbar.update(1)
        pbar.close()

        #### LCPL starts here
        self.bg_type = 'LCPL'
        y_fit = mean_spec       
        bg_func, d_func, p0, bounds = self.prepare_bgfit( y_fit, e_fit )
        ## Get (5, 95)% percentile of exponents
        r1s = np.zeros( (self.ny, self.nx) )
        r2s = np.zeros( (self.ny, self.nx) )
        A1s = np.zeros( (self.ny, self.nx) )
        A2s = np.zeros( (self.ny, self.nx) )

        ## Get (5, 95)% percentile of exponents
        r_0595 = np.percentile( rs, (0,95) )

        p0 = (p0[0], r_0595[0], p0[2], r_0595[1])

        bounds = ([     0, 0,      0, 0], 
                  [np.inf, 10, np.inf, 10])


        ## Fit LCPL
        pbar = tqdm_notebook(total = (self.nx)*(self.ny),desc = "LCPL Background subtracting")
        for i in range(self.ny):
            for j in range(self.nx):
                y_fit = si_fit[i,j]
                
                
                # try:
                p_fit, cov_fit = curve_fit( bg_func, e_fit, y_fit, p0=p0, bounds=bounds,
                                           method='trf',jac=d_func, ftol=ftol, gtol=gtol, maxfev=5000)
            
                bg_fit = bg_func(self.es, *p_fit)
                # except:
                #     bg_fit = 0

                bsub = self.si[i,j]-bg_fit
                bsub[0:fit_i_ch] = 0
                bg_pl_SI[i,j,:] = bsub

                A1s[i,j] = p_fit[0]
                r1s[i,j] = p_fit[1]
                A2s[i,j] = p_fit[2]
                r2s[i,j] = p_fit[3]
                pbar.update(1)
        pbar.close()

        if plot_bg_coef:
            fig,ax=plt.subplots(3,2)
            fig.suptitle( 'LCPL: A1*E^(-r1)+A2*E^(-r2)' )
            ax[0,1].matshow(r1s)
            ax[0,1].set_title('r1')
            ax[1,1].matshow(r2s)
            ax[1,1].set_title('r2')
            ax[0,0].matshow(A1s)
            ax[0,0].set_title('A1')
            ax[1,0].matshow(A2s)
            ax[1,0].set_title('A2')

            ax[2,0].matshow(As)
            ax[2,0].set_title('Powerlaw Guess (A)')
            ax[2,1].matshow(rs)
            ax[2,0].set_title('Powerlaw Guess (r)')
                
        return bg_pl_SI, self.e_bsub
    
    def bgsub_LCPL_Fixed_SI( self, e_bsub=None, lba=False, LCPL_percentile=(5,95), plot_bg_coef=False, lba_gfwhm=5, ftol=0.0005, gtol=0.00005):#,xtol=None):
        if e_bsub is not None:
            self.e_bsub = e_bsub


        #### Do standard Powerlaw fitting to generate guesses
        self.bg_type = 'powerlaw'
        bg_pl_SI = np.zeros_like(self.si)
        fit_i_ch = self.eVtoCh(self.e_bsub[0], self.es)
        fit_f_ch = self.eVtoCh(self.e_bsub[1], self.es)
        e_fit = self.es[fit_i_ch:fit_f_ch]

        #### Setup for LBA if requested
        if lba :
            si_fit = self.prepare_lba( lba_gfwhm=lba_gfwhm )
        else :
            si_fit = self.si.copy()[:,:,fit_i_ch:fit_f_ch]
    
        ## Fit Mean Spectrum to get a good guess Parameter
        mean_spec = np.mean( si_fit, axis=(0,1))
        y_fit = mean_spec        
        bg_func, d_func, p0, bounds = self.prepare_bgfit( y_fit, e_fit )
        
        ## perform PL fit on each pixel.
        rs = np.zeros( (self.ny, self.nx) )
        As = np.zeros( (self.ny, self.nx) )
        pbar = tqdm_notebook(total = (self.nx)*(self.ny),desc = "Fitting Normal Powerlaw")
        for i in range(self.ny):
            for j in range(self.nx):
                y_fit = si_fit[i,j]
                
                p_fit, cov_fit = curve_fit( bg_func, e_fit, y_fit, p0=p0, bounds=bounds,
                                           method='trf',jac=d_func, ftol=ftol, gtol=gtol, maxfev=5000)
            
                As[i,j] = p_fit[0]
                rs[i,j] = p_fit[1]
                pbar.update(1)
        pbar.close()

        #### LCPL starts here
        self.bg_type = 'LCPL'
        y_fit = mean_spec       
        bg_func, d_func, p0, bounds = self.prepare_bgfit( y_fit, e_fit )
        ## Get (5, 95)% percentile of exponents
        r_0595 = np.percentile( rs, LCPL_percentile )
        r1s = np.zeros( (self.ny, self.nx) )
        r2s = np.zeros( (self.ny, self.nx) )
        A1s = np.zeros( (self.ny, self.nx) )
        A2s = np.zeros( (self.ny, self.nx) )

        p0 = (p0[0], r_0595[0], p0[2], r_0595[1])

        bounds = ([     0, r_0595[0]*0.97,      0, r_0595[1]*0.97], 
                  [np.inf, r_0595[0]*1.03, np.inf, r_0595[1]*1.03])


        ## Fit LCPL
        pbar = tqdm_notebook(total = (self.nx)*(self.ny),desc = "LCPL Background subtracting")
        for i in range(self.ny):
            for j in range(self.nx):
                y_fit = si_fit[i,j]
                
                # try:
                p_fit, cov_fit = curve_fit( bg_func, e_fit, y_fit, p0=p0, bounds=bounds,
                                           method='trf',jac=d_func, ftol=ftol, gtol=gtol, maxfev=5000)
            
                bg_fit = bg_func(self.es, *p_fit)
                # except:
                #     bg_fit = 0

                bsub = self.si[i,j]-bg_fit
                bsub[0:fit_i_ch] = 0
                bg_pl_SI[i,j,:] = bsub

                A1s[i,j] = p_fit[0]
                r1s[i,j] = p_fit[1]
                A2s[i,j] = p_fit[2]
                r2s[i,j] = p_fit[3]
                pbar.update(1)
        pbar.close()

        if plot_bg_coef:
            fig,ax=plt.subplots(3,2)
            fig.suptitle( 'LCPL: A1*E^(-r1)+A2*E^(-r2)' )
            ax[0,1].matshow(r1s)
            ax[0,1].set_title('r1')
            ax[1,1].matshow(r2s)
            ax[1,1].set_title('r2')
            ax[0,0].matshow(A1s)
            ax[0,0].set_title('A1')
            ax[1,0].matshow(A2s)
            ax[1,0].set_title('A2')

            ax[2,0].matshow(As)
            ax[2,0].set_title('Powerlaw Guess (A)')
            ax[2,1].matshow(rs)
            ax[2,0].set_title('Powerlaw Guess (r)')
                
        return bg_pl_SI, self.e_bsub

    def prepare_lba( self, e_bsub=None, lba_gfwhm=5 ):
        if e_bsub is not None:
            self.e_bsub = e_bsub
        self.gwfm = lba_gfwhm

        fit_i_ch = self.eVtoCh( self.e_bsub[0], self.es)
        fit_f_ch = self.eVtoCh( self.e_bsub[1], self.es)


        lba_raw = np.copy( self.si )
        for e_ch in np.arange( fit_i_ch, fit_f_ch ):
            lba_raw[:,:,e_ch] = gaussian_filter( self.si[:,:,e_ch], sigma=lba_gfwhm/2.35)

        lba_normalized = np.copy(lba_raw)
        pbar = tqdm_notebook(total = (self.nx * self.ny ),desc = "LBA Normalizing")
        for i in range(self.ny):
            for j in range(self.nx):
                normalize,pcov_pl=curve_fit( self.lba_normalization,
                                            lba_raw[i,j,fit_i_ch:fit_f_ch],
                                            self.si[i,j,fit_i_ch:fit_f_ch])
                lba_normalized[i,j,fit_i_ch:fit_f_ch] = lba_raw[i,j,fit_i_ch:fit_f_ch]*normalize
                pbar.update(1)
        pbar.close()

        return lba_normalized[:,:,fit_i_ch:fit_f_ch]
    
    def lba_normalization( self, lba, m ):
        return m*lba
    
    def dummy( self,a,b):
        ""

    def eVtoCh(self, energy, array):
        return int(np.squeeze(np.argwhere(array == self.find_nearest(array,energy))))
    def ChtoeV(self, channel, array):
        return array[channel]
    def find_nearest(self, array, value):
        """
        Inputs: 
        array - array... 
        value - value to search for in array
        
        Outputs:
        array[idx] - nearest value in array
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
        

def bgsub_1D(raw_data, energy_axis, edge, **kwargs):
    """
    Full background subtraction function for the 1D case-
    Optional LBA, log fitting, LCPL, and exponential fitting.
    For more information on non-linear fitting function, see information at https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    Inputs:
    raw_data - 1D spectrum
    energy_axis - corresponding energy axis
    edge - edge parameters defined by KEM convention

    **kawrgs:
    fit - choose the type of background fit, default == 'pl' == Power law. Can also use 'exp'== Exponential, 'lin' == Linear, 'lcpl' == LCPL.
    log - Boolean, if true, log transform data and fit using QR factorization, default == False.
    nstd - Standard deviation spread of r error from non-linear power law fitting. Default == 100.
    ftol - default to 0.0005, Relative error desired in the sum of squares.
    gtol - default to 0.00005, Orthogonality desired between the function vector and the columns of the Jacobian.
    xtol - default to None, Relative error desired in the approximate solution.
    maxfev - default to 50000, Only change if you are consistenly catching runtime errors and loosening gtol/ftols are not making a good enough fit.
    method - default is 'trf', see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares for description of methods
    Note: may need stricter tolerances on ftol/gtol for noisier data. Anecdotally, a stricter gtol (as low as 1e-8) has a larger effect on the quality of the bgsub.

    Outputs:
    bg_1D - background spectrum
    """
    fit_start_ch = eVtoCh(edge[0], energy_axis)
    fit_end_ch = eVtoCh(edge[1], energy_axis)
    zdim = len(raw_data)
    ewin = energy_axis[fit_start_ch:fit_end_ch]
    esub = energy_axis[fit_start_ch:]
    bg_1D = np.zeros_like(raw_data)
    fy = np.zeros((1,zdim))
    fy[0,:] = raw_data


    """
            elif fit == 'lcpl':
                fitfunc = lcpowerlaw
    """
    """
## Either fast fitting -> log fitting, Or slow fitting -> non-linear fitting
    if 'log' in kwargs.keys():
        log = kwargs['log']
    else:
        log = True

## Fitting parameters for non-linear curve fitting if non-log based fitting
        if 'ftol' in kwargs.keys():
            ftol = kwargs['ftol']
        else:
            ftol = 1e-8
        if 'gtol' in kwargs.keys():
            gtol = kwargs['gtol']
        else:
            gtol = 1e-8
        if 'xtol' in kwargs.keys():
            xtol = kwargs['xtol']
        else:
            xtol = 1e-8
        if 'maxfev' in kwargs.keys():
            maxfev = kwargs['maxfev']
        else:
            maxfev = 50000
        if 'method' in kwargs.keys():
            method = kwargs['method']
        else:
            method = 'trf'
    """
## Determine if fitting is power law or exponenetial
    if 'fit' in kwargs.keys():
        fit = kwargs['fit']
        if fit == 'exp':
            fitfunc = exponential
        elif fit == 'pl':
            fitfunc = powerlaw
        elif fit == 'lin':
            fitfunc = linear
        else:
            print('Did not except fitting function, please use either \'pl\' for powerlaw, \'exp\' for exponential, \'lin\' for linear or \'lcpl\' for LCPL.')
    else:
        fitfunc = powerlaw



## If fast fitting linear background, find fit using qr factorization
    if fitfunc==linear:
        Blin = fy[:,fit_start_ch:fit_end_ch]
        Alin = np.zeros((len(ewin),2))
        Alin[:,0] = np.ones(len(ewin))
        Alin[:,1] = ewin
        Xlin = qrnorm(Alin,Blin.T)
        Elin = np.zeros((len(esub),2))
        Elin[:,0] = np.ones(len(esub))
        Elin[:,1] = esub
        bgndLINline = np.dot(Xlin.T,Elin.T)
        bg_1D[fit_start_ch:] = raw_data[fit_start_ch:] - bgndLINline
        rval = np.squeeze(Xlin[1,:])

## If fast log fitting and powerlaw, find fit using qr factorization
    elif fitfunc==powerlaw:
        Blog = fy[:,fit_start_ch:fit_end_ch]
        Alog = np.zeros((len(ewin),2))
        Alog[:,0] = np.ones(len(ewin))
        Alog[:,1] = np.log(ewin)
        Xlog = qrnorm(Alog,np.log(abs(Blog.T)))
        Elog = np.zeros((len(esub),2))
        Elog[:,0] = np.ones(len(esub))
        Elog[:,1] = np.log(esub)
        bgndPLline = np.exp(np.dot(Xlog.T,Elog.T))
        bg_1D[fit_start_ch:] = raw_data[fit_start_ch:] - bgndPLline
        rval = np.squeeze(Xlog[1,:])

## If fast log fitting and exponential, find fit using qr factorization
    elif fitfunc==exponential:
        Bexp = fy[:,fit_start_ch:fit_end_ch]
        Aexp = np.zeros((len(ewin),2))
        Aexp[:,0] = np.ones(len(ewin))
        Aexp[:,1] = ewin
        Xexp = qrnorm(Aexp,np.log(abs(Bexp.T)))
        Eexp = np.zeros((len(esub),2))
        Eexp[:,0] = np.ones(len(esub))
        Eexp[:,1] = esub
        bgndEXPline = np.exp(np.dot(Xexp.T,Eexp.T))
        bg_1D[fit_start_ch:] = raw_data[fit_start_ch:] - bgndEXPline
        rval = np.squeeze(Xexp[1,:])
    """



## Power law non-linear curve fitting using scipy.optimize.curve_fit
    elif ~log & (fitfunc==powerlaw):
        popt_pl,pcov_pl=curve_fit(powerlaw, ewin, raw_data[fit_start_ch:fit_end_ch],maxfev=maxfev,method=method,
                                  verbose = 0, ftol=ftol, gtol=gtol, xtol=xtol)
        c,r = popt_pl
        bg_1D[fit_start_ch:] = raw_data[fit_start_ch:] - powerlaw(energy_axis[fit_start_ch:],c,r)

## Exponential non-linear curve fitting using scipy.optimize.curve_fit
    elif ~log & (fitfunc==exponential):
        popt_exp,pcov_exp=curve_fit(exponential, ewin, raw_data[fit_start_ch:fit_end_ch],maxfev=maxfev,method=method,
                                    verbose = 0,p0=[0,0], ftol=ftol, gtol=gtol, xtol=xtol)
        a,b = popt_exp
        bg_1D[fit_start_ch:] = raw_data[fit_start_ch:] - exponential(energy_axis[fit_start_ch:],a,b)

## LCPL non-linear curve fitting using scipy.optimize.curve_fit
    elif fitfunc==lcpowerlaw:
        if 'nstd' in kwargs.keys():
            nstd = kwargs['nstd']
        else:
            nstd = 100
        popt_pl,pcov_pl=curve_fit(powerlaw, ewin, raw_data[fit_start_ch:fit_end_ch],maxfev=maxfev,method=method,
                                  verbose = 0, ftol=ftol, gtol=gtol, xtol=xtol)
        c,r = popt_pl
        perr = np.sqrt(np.diag(pcov_pl))
        rstd = perr[1]
        popt_lcpl,pcov_lcpl=curve_fit(lcpowerlaw, ewin, raw_data[fit_start_ch:fit_end_ch],maxfev=maxfev,method=method,
                                    verbose = 0,p0=[c/2,r-nstd*rstd,c/2,r+nstd*rstd], ftol=ftol, gtol=gtol, xtol=xtol)
        c1,r1,c2,r2 = popt_lcpl
        bg_1D[fit_start_ch:] = raw_data[fit_start_ch:] - lcpowerlaw(energy_axis[fit_start_ch:],c1,r1,c2,r2)
    """
    return bg_1D,rval
 


def bgsub_SI(raw_data, energy_axis, edge, **kwargs):
    """
    Full background subtraction function-
    Optional LBA, log fitting, LCPL, and exponential fitting.
    For more information on non-linear fitting function, see information at https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    Inputs:
    raw_data - SI
    energy_axis - corresponding energy axis
    edge - edge parameters defined by KEM convention

    **kawrgs:
    fit - choose the type of background fit, default == 'pl' == Power law. Can also use 'exp'== Exponential, 'lin' == Linear.
    gfwhm - If using LBA, gfwhm corresponds to width of gaussian filter, default = None, meaning no LBA.
    log - Boolean, if true, log transform data and fit using QR factorization, default == False.
    lc - Boolean, if true, include LCPL or LCEX background subtracted SI, default == False.
    nstd - standard deviation spread of r values from power law fitting. 1= 20/80 percentile, 2= 5/95 percentile. Default == 2.
    ftol - default to 0.0005, Relative error desired in the sum of squares.
    gtol - default to 0.00005, Orthogonality desired between the function vector and the columns of the Jacobian.
    xtol - default to None, Relative error desired in the approximate solution.
    maxfev - default to 50000, Only change if you are consistenly catching runtime errors and loosening gtol/ftols are not making a good enough fit.
    method - default is 'trf', see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares for description of methods
    Note: may need stricter tolerances on ftol/gtol for noisier data. Anecdotally, a stricter gtol (as low as 1e-8) has a larger effect on the quality of the bgsub.
    mask - Boolean mask defines non-vacuum region in SI, used to improve LCPL
    threshold - mininum average counts in fit window to be included in LCPL calculation.

    Outputs:
    if lcpl == False:
        bg_pl_SI - background subtracted SI
    if lcpl == True:
        bg_pl_SI, bg_lcpl_SI - background subtracted SI, LCPL background subtracted SI
    """
    fit_start_ch = eVtoCh(edge[0], energy_axis)
    fit_end_ch = eVtoCh(edge[1], energy_axis)
    raw_data = raw_data.astype('float32')
    if len(np.shape(raw_data)) == 2:
        tempx,tempz = np.shape(raw_data)
        raw_data = np.reshape(raw_data,(tempx,1,tempz))
    if len(np.shape(raw_data)) == 1:
        tempz = len(raw_data)
        raw_data = np.reshape(raw_data,(1,1,tempz))
    xdim, ydim, zdim = np.shape(raw_data)
    ewin = energy_axis[fit_start_ch:fit_end_ch]
    esub = energy_axis[fit_start_ch:]
    bg_pl_SI = np.zeros_like(raw_data)

## Special case: if there is vacuum in the SI and it is causing trouble with your LCPL fitting:
    if 'mask' in kwargs.keys():
        threshmask = kwargs['mask']
    elif 'threshold' in kwargs.keys():
        thresh = kwargs['threshold']
        mean_back = np.mean(raw_data[:,:,fit_start_ch:fit_end_ch],axis=2)
        threshmask = mean_back > thresh
    else:
        mask = np.ones((xdim,ydim))
        threshmask = mask == 1

    if 'gfwhm' in kwargs.keys():
        gfwhm = kwargs['gfwhm']
        lba_raw = np.copy(raw_data)
        lba_raw_normalized = np.copy(lba_raw)
        for energychannel in np.arange(fit_start_ch,fit_end_ch):
            lba_raw[:,:,energychannel] = gaussian_filter(raw_data[:,:,energychannel],sigma=gfwhm/2.35)
        pbar = tqdm(total = (xdim)*(ydim),desc = "Normalizing")
        for i in range(xdim):
            for j in range(ydim):
                lba_mean = np.mean(lba_raw[i,j,fit_start_ch:fit_end_ch])
                data_mean = np.mean(raw_data[i,j,fit_start_ch:fit_end_ch])
                lba_raw_normalized[i,j,fit_start_ch:fit_end_ch] = lba_raw[i,j,fit_start_ch:fit_end_ch]*data_mean/lba_mean
                pbar.update(1)
    else:
        lba_raw_normalized = np.copy(raw_data)

## Either fast fitting -> log fitting, Or slow fitting -> non-linear fitting
    if 'log' in kwargs.keys():
        log = kwargs['log']
    else:
        log = False

## Fitting parameters for non-linear curve fitting if non-log based fitting
    if 'ftol' in kwargs.keys():
        ftol = kwargs['ftol']
    else:
        ftol = 0.0005
    if 'gtol' in kwargs.keys():
        gtol = kwargs['gtol']
    else:
        gtol = 0.00005
    if 'xtol' in kwargs.keys():
        xtol = kwargs['xtol']
    else:
        xtol = None
    if 'maxfev' in kwargs.keys():
        maxfev = kwargs['maxfev']
    else:
        maxfev = 50000
    if 'method' in kwargs.keys():
        method = kwargs['method']
    else:
        method = 'trf'

## Determine if fitting is power law or exponenetial
    if 'fit' in kwargs.keys():
        fit = kwargs['fit']
        if fit == 'exp':
            fitfunc = exponential
            bounds = ([0, 0], [np.inf, np.inf])
        elif fit == 'pl':
            fitfunc = powerlaw
        elif fit == 'lin':
            fitfunc = linear
        else:
            print('Did not except fitting function, please use either \'pl\' for powerlaw, \'exp\' for exponential or \'lin\' for linear.')
    else:
        fitfunc = powerlaw

## If fast fitting linear background, find fit using qr factorization
    if fitfunc==linear:
        Blin = np.reshape(lba_raw_normalized[:,:,fit_start_ch:fit_end_ch],((xdim*ydim),len(ewin)))
        Alin = np.zeros((len(ewin),2))
        Alin[:,0] = np.ones(len(ewin))
        Alin[:,1] = ewin
        Xlin = qrnorm(Alin,Blin.T)
        Elin = np.zeros((len(esub),2))
        Elin[:,0] = np.ones(len(esub))
        Elin[:,1] = esub
        bgndLINline = np.dot(Xlin.T,Elin.T)
        bgndLIN = np.reshape(bgndLINline,(xdim,ydim,len(esub)))
        bg_pl_SI[:,:,fit_start_ch:] = raw_data[:,:,fit_start_ch:] - bgndLIN

## If fast log fitting and powerlaw, find fit using qr factorization
    if log & (fitfunc==powerlaw):
        Blog = np.reshape(lba_raw_normalized[:,:,fit_start_ch:fit_end_ch],((xdim*ydim),len(ewin)))
        Alog = np.zeros((len(ewin),2))
        Alog[:,0] = np.ones(len(ewin))
        Alog[:,1] = np.log(ewin)
        Xlog = qrnorm(Alog,np.log(abs(Blog.T)))
        Elog = np.zeros((len(esub),2))
        Elog[:,0] = np.ones(len(esub))
        Elog[:,1] = np.log(esub)
        bgndPLline = np.exp(np.dot(Xlog.T,Elog.T))
        bgndPL = np.reshape(bgndPLline,(xdim,ydim,len(esub)))
        bg_pl_SI[:,:,fit_start_ch:] = raw_data[:,:,fit_start_ch:] - bgndPL
        maskline = np.reshape(threshmask,(xdim*ydim))
        rline_long = -1*Xlog[1,:]
        rline = rline_long[maskline]

## If fast log fitting and exponential, find fit using qr factorization
    elif log & (fitfunc==exponential):
        Bexp = np.reshape(lba_raw_normalized[:,:,fit_start_ch:fit_end_ch],((xdim*ydim),len(ewin)))
        Aexp = np.zeros((len(ewin),2))
        Aexp[:,0] = np.ones(len(ewin))
        Aexp[:,1] = ewin
        Xexp = qrnorm(Aexp,np.log(abs(Bexp.T)))
        Eexp = np.zeros((len(esub),2))
        Eexp[:,0] = np.ones(len(esub))
        Eexp[:,1] = esub
        bgndEXPline = np.exp(np.dot(Xexp.T,Eexp.T))
        bgndEXP = np.reshape(bgndEXPline,(xdim,ydim,len(esub)))
        bg_pl_SI[:,:,fit_start_ch:] = raw_data[:,:,fit_start_ch:] - bgndEXP
        maskline = np.reshape(threshmask,(xdim*ydim))
        bline_long = -1*Xexp[1,:]
        bline = bline_long[maskline]

## Power law non-linear curve fitting using scipy.optimize.curve_fit
    elif ~log & (fitfunc==powerlaw):
        rline = []
        dummyspec = sum(sum(raw_data))/(xdim*ydim)
        popt_init,pcov_init=curve_fit(powerlaw, ewin, dummyspec[fit_start_ch:fit_end_ch],maxfev=maxfev,method=method,verbose = 0)
        pbar1 = tqdm(total = (xdim)*(ydim),desc = "Background subtracting")
        for i in range(xdim):
            for j in range(ydim):
                popt_pl,pcov_pl=curve_fit(powerlaw, ewin, lba_raw_normalized[i,j,fit_start_ch:fit_end_ch],maxfev=maxfev,method=method,verbose = 0
                                          ,p0=popt_init, ftol=ftol, gtol=gtol, xtol=xtol)
                c,r = popt_pl
                if threshmask[i,j]:
                    rline = np.append(rline,r)
                bg_pl_SI[i,j,fit_start_ch:] = raw_data[i,j,fit_start_ch:] - powerlaw(energy_axis[fit_start_ch:],c,r)
                pbar1.update(1)

## Exponential non-linear curve fitting using scipy.optimize.curve_fit
    elif ~log & (fitfunc==exponential):
        bline = []
        # dummyspec = sum(sum(raw_data))/(xdim*ydim)
        # popt_init,pcov_init=curve_fit(exponential, ewin, dummyspec[fit_start_ch:fit_end_ch],bounds=bounds,p0=[0,0],maxfev=maxfev,method=method,verbose = 0)
        pbar1 = tqdm(total = (xdim)*(ydim),desc = "Background subtracting")
        for i in range(xdim):
            for j in range(ydim):
                popt_exp,pcov_exp=curve_fit(exponential, ewin, lba_raw_normalized[i,j,fit_start_ch:fit_end_ch],maxfev=maxfev,method=method,verbose = 0
                                          ,p0=[0,0], ftol=ftol, gtol=gtol, xtol=xtol)
                a,b = popt_exp
                if threshmask[i,j]:
                    bline = np.append(bline,b)
                bg_pl_SI[i,j,fit_start_ch:] = raw_data[i,j,fit_start_ch:] - exponential(energy_axis[fit_start_ch:],a,b)
                pbar1.update(1)

## Given r values of SI, refit background using a linear combination of power laws, using either 5/95 percentile or 20/80 percentile r values.
    if 'lc' in kwargs.keys():
        lc = kwargs['lc']
    else:
        lc = False

    if lc & (fitfunc==powerlaw):
        if 'nstd' in kwargs.keys():
            nstd = kwargs['nstd']
        else:
            nstd = 2
        bg_lcpl_SI = np.zeros_like(raw_data)
        rmu,rstd = norm.fit(rline)
        rmin = rmu - nstd*rstd
        rmax = rmu + nstd*rstd
        if nstd == 2:
            print('5th percentile power law = {}'.format(rmin))
            print('95th percentile power law = {}'.format(rmax))
        elif nstd == 1:
            print('20th percentile power law = {}'.format(rmin))
            print('80th percentile power law = {}'.format(rmax))
        else:
            print('Min power law = {}'.format(rmin))
            print('Max power law = {}'.format(rmax))
        B = np.reshape(lba_raw_normalized[:,:,fit_start_ch:fit_end_ch],((xdim*ydim),len(ewin)))
        A = np.zeros((len(ewin),2))
        A[:,0] = ewin**(-rmin)
        A[:,1] = ewin**(-rmax)
        X = qrnorm(A,B.T)
        E = np.zeros((len(esub),2))
        E[:,0] = esub**(-rmin)
        E[:,1] = esub**(-rmax)
        bgndLCPLline = np.dot(X.T,E.T)
        bgndLCPL = np.reshape(bgndLCPLline,(xdim,ydim,len(esub)))
        bg_lcpl_SI[:,:,fit_start_ch:] = raw_data[:,:,fit_start_ch:] - bgndLCPL
        return bg_pl_SI, bg_lcpl_SI

### Testing
    elif lc & (fitfunc==exponential):
        if 'nstd' in kwargs.keys():
            nstd = kwargs['nstd']
        else:
            nstd = 2
        bg_lcpl_SI = np.zeros_like(raw_data)
        bmu,bstd = norm.fit(bline)
        bmin = bmu - nstd*bstd
        bmax = bmu + nstd*bstd
        if nstd == 2:
            print('5th percentile exponential = {}'.format(bmin))
            print('95th percentile exponential = {}'.format(bmax))
        elif nstd == 1:
            print('20th percentile exponential = {}'.format(bmin))
            print('80th percentile exponential = {}'.format(bmax))
        else:
            print('Min exponential = {}'.format(bmin))
            print('Max exponential = {}'.format(bmax))
        B = np.reshape(lba_raw_normalized[:,:,fit_start_ch:fit_end_ch],((xdim*ydim),len(ewin)))
        A = np.zeros((len(ewin),2))
        A[:,0] = np.exp(-bmin*ewin)
        A[:,1] = np.exp(-bmax*ewin)
        X = qrnorm(A,B.T)
        E = np.zeros((len(esub),2))
        E[:,0] = np.exp(-bmin*esub)
        E[:,1] = np.exp(-bmax*esub)
        bgndLCPLline = np.dot(X.T,E.T)
        bgndLCPL = np.reshape(bgndLCPLline,(xdim,ydim,len(esub)))
        bg_lcpl_SI[:,:,fit_start_ch:] = raw_data[:,:,fit_start_ch:] - bgndLCPL
        return bg_pl_SI, bg_lcpl_SI

    else:
        return bg_pl_SI
    

def bgsub_fast(raw_data, energy_axis, fit_window, rval,fit='pl'):

    fit_start_ch = eVtoCh(fit_window[0], energy_axis)
    fit_end_ch = eVtoCh(fit_window[1], energy_axis)
    raw_data = raw_data.astype('float32')

    xdim, ydim, zdim = np.shape(raw_data)
    ewin = energy_axis[fit_start_ch:fit_end_ch]
    esub = energy_axis[fit_start_ch:]
    bg_SI = np.zeros_like(raw_data)  

    if fit == 'lin':
        B = np.reshape(raw_data[:,:,fit_start_ch:fit_end_ch],((xdim*ydim),len(ewin)))
        A = np.zeros((len(ewin),1))
        A[:,0] = ewin*(rval)
        X = qrnorm(A,B.T)
        E = np.zeros((len(esub),1))
        E[:,0] = esub*(rval)

        bgndPLline = np.dot(X.T,E.T)
        bgndPL = np.reshape(bgndPLline,(xdim,ydim,len(esub)))
        bg_SI[:,:,fit_start_ch:] = raw_data[:,:,fit_start_ch:] - bgndPL

    if fit == 'pl':
        B = np.reshape(raw_data[:,:,fit_start_ch:fit_end_ch],((xdim*ydim),len(ewin)))
        A = np.zeros((len(ewin),1))
        A[:,0] = ewin**(rval)
        X = qrnorm(A,B.T)
        E = np.zeros((len(esub),1))
        E[:,0] = esub**(rval)

        bgndPLline = np.dot(X.T,E.T)
        bgndPL = np.reshape(bgndPLline,(xdim,ydim,len(esub)))
        bg_SI[:,:,fit_start_ch:] = raw_data[:,:,fit_start_ch:] - bgndPL

    if fit == 'exp':
        B = np.reshape(raw_data[:,:,fit_start_ch:fit_end_ch],((xdim*ydim),len(ewin)))
        A = np.zeros((len(ewin),1))
        A[:,0] = np.exp(ewin*(rval))
        X = qrnorm(A,B.T)
        E = np.zeros((len(esub),1))
        E[:,0] = np.exp(esub*(rval))

        bgndPLline = np.dot(X.T,E.T)
        bgndPL = np.reshape(bgndPLline,(xdim,ydim,len(esub)))
        bg_SI[:,:,fit_start_ch:] = raw_data[:,:,fit_start_ch:] - bgndPL

    return bg_SI


def eVtoCh(energy, array):
	return int(np.squeeze(np.argwhere(array == find_nearest(array,energy))))


def find_nearest(array, value):
    """
    Inputs:
    array - array...
    value - value to search for in array

    Outputs:
    array[idx] - nearest value in array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def qrnorm(A,b):
    """
    Solve systems of linear equations Ax = b for x
    """
    q, r = LA.qr(A)
    p = np.dot(q.T,b)
    return np.dot(LA.inv(r),p)



def linear(energy, a, b):
    return a*energy + b

def powerlaw(energy, c, r):
    return c*energy**(-r)

def lcpowerlaw(energy, c1, r1, c2, r2):
    return c1*energy**(-r1) + c2*energy**(-r2)

def exponential(energy,a,b):
    return a*np.exp(-b*energy)


class SIbrowser:
    
    def __init__( self, si, energy, im_adf=None, cmap='gray', figsize=(9,4), **kwargs):
        ######### Initialize browser object ##########
        self.si = si
        self.energy = energy
        self.im_adf = im_adf
        self.im_inel =np.mean(si,axis=(-1))
        self.edge = eels_edge( "", e_bsub=None, e_int=(energy[0], energy[-1]))

        self.spectrum1=np.mean(si,axis=(0,1))
        self.spectrum2=np.mean(si,axis=(0,1))
        
        ##############Initialize Display#################
        self.fig=plt.figure(figsize=figsize)

        self.ax= {'inel':None,'spec':None,'bttn':None}
        self.ax['inel']=self.fig.add_axes([0.05,0.3,0.425,0.6]) # Image
        self.ax['spec']=self.fig.add_axes([0.55,0.3,0.425,0.6]) # Spectrum
        self.ax['bttn']=self.fig.add_axes([0.88,0.012,0.1,0.2]) # Button
        self.ax['bttn'].axis('off')
        ##############################################
        # Initialize plot handles
        self.h = {'inel':None, 'spec1':None, 'spec2':None}

        ## Inelastic Image
        self.h['inel']  = self.ax['inel'].matshow(self.im_inel,cmap = cmap)
        self.ax['inel'].set_axis_off()
        self.ax['inel'].set_title('Inelastic image')
        ## Spectra
        self.h['spec1'], =self.ax['spec'].plot(self.energy,self.spectrum1,color='maroon')
        self.h['spec2'], =self.ax['spec'].plot(self.energy,self.spectrum2,color='k',alpha=0)

        self.ax['spec'].set_ylim([self.spectrum1.min(),self.spectrum1.max()])
        self.ax['spec'].set_xlim([self.energy.min(),self.energy.max()])
        self.ax['spec'].set_yticks([])
        self.ax['spec'].set_xlabel('Energy (keV)')
        self.ax['spec'].set_ylabel('Intensity')
        self.ax['spec'].set_title('EELS spectrum')
        ##############################################
        results_dict={}
        for key in ['spectrum','image','roi','energy_span']:
            results_dict[key]=[]
        
        ################### Selectors ###################
        self.ui = {'roi1':None, 'roi2':None, 'spec':None, 'logscale':None}
        self.ui['roi1'] = RectangleSelector(self.ax['inel'], self.dummy, button=[1],
                                        useblit=False ,minspanx=1, minspany=1,spancoords='pixels',
                                        interactive=True,props=dict(facecolor='crimson',edgecolor='crimson',alpha=0.2,fill=True),
                                        handle_props=dict(markersize=2,markerfacecolor='white'))#,ignore_event_outside=True
        
        self.ui['roi2'] = RectangleSelector(self.ax['inel'], self.dummy, button=[3],
                                        useblit=False ,minspanx=1, minspany=1,spancoords='pixels',
                                        interactive=True,props=dict(facecolor='royalblue',edgecolor='royalblue',alpha=0.2,fill=True),
                                        handle_props=dict(markersize=2,markerfacecolor='white'))#,ignore_event_outside=True)
            
        self.ui['span_spec'] = SpanSelector(self.ax['spec'], self.dummy, button=[1],
                                            useblit=False, minspan=1,direction="horizontal",
                                            interactive=True,props=dict(facecolor='green',edgecolor='green',alpha=0.2,fill=True),
                                            grab_range=10, drag_from_anywhere=True)
        

        self.ui['logscale']=CheckButtons(self.ax['bttn'],["Log Scale"],useblit=True ,)
        self.ui['logscale'].on_clicked(self.scale_button)

        self.fig.canvas.mpl_connect( 'motion_notify_event', 
                                    lambda event: self.onclick_figure(event))

    ############### Event Handlers ###################
    def onclick_figure( self, event ):
        if event.inaxes in [self.ax['inel']]:
            if event.button == MouseButton.LEFT:
                # Left Click on Inelastic Image
                real_roi1 = np.array(self.ui['roi1'].extents).astype('int')
                self.spectrum1=np.mean(self.si[int(real_roi1[2]):int(real_roi1[3]),int(real_roi1[0]):int(real_roi1[1]),:],axis=(0,1))
                self.update_spectrum1()

            elif event.button == MouseButton.RIGHT:
                # Right Click on Inelastic Image
                real_roi2 = np.array(self.ui['roi2'].extents).astype('int')
                self.spectrum2=np.mean(self.si[int(real_roi2[2]):int(real_roi2[3]),int(real_roi2[0]):int(real_roi2[1]),:],axis=(0,1))
                self.update_spectrum2()
        elif event.inaxes in [self.ax['spec']]:
            if event.button == MouseButton.LEFT:
                self.edge.e_int = self.ui['span_spec'].extents
                self.update_image()

    ################### Update Functions ###################
    def update_spectrum1(self):
        self.h['spec1'].set_ydata(self.spectrum1)
        self.h['spec1'].set_color('maroon')
        try:
            self.ax['spec'].set_ylim([min(self.spectrum1.min(),self.spectrum2.min()),max(self.spectrum1.max(),self.spectrum2.max())])
        except:
            pass
        
    def update_spectrum2(self):
        self.h['spec2'].set_ydata(self.spectrum2)
        self.h['spec2'].set_alpha(1)
        self.h['spec2'].set_color('cadetblue')
        try:
            self.ax['spec'].set_ylim([min(self.spectrum1.min(),self.spectrum2.min()),max(self.spectrum1.max(),self.spectrum2.max())])
        except:
            pass

    def update_image(self):
        indmin, indmax = np.searchsorted(self.energy, self.edge.e_int)
        self.im_inel = np.mean(self.si[:,:,indmin:indmax],axis=(-1))
        self.h['inel'].set_data(self.im_inel)
        self.h['inel'].autoscale()

        
    def scale_button(self, event):
        if self.ax['spec'].get_yscale()=='linear':
            
            self.ax['spec'].set_yscale('log')
            self.ax['spec'].set_ylabel('Log Intensity')
            self.ax['spec'].set_yticks([])
        else:
            
            self.ax['spec'].set_yscale('linear')
            self.ax['spec'].set_ylabel('Intensity')
            self.ax['spec'].set_yticks([])

    def dummy( self, *args ):
        pass