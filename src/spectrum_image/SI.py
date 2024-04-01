import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, RectangleSelector, SpanSelector, CheckButtons, RadioButtons, Button
from matplotlib.backend_bases import MouseButton
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, tqdm_notebook
import spectrum_image.SI_lineshapes as ls
from scipy.stats import norm

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



class SI :
    def __init__( self, si, energy, ADF=[] ):

        si[np.isnan(si)] = 0
        self.si = si
        (self.ny, self.nx, self.ne) = self.si.shape
        
        # 

        self.energy = np.asarray( energy )
        
    def fitbrowser( self, edge=None, cmap='gray', figsize=(9,6), lc=False, gfwhm=10, log=True, 
                 ftol=0.0005, gtol=0.00005, xtol=None, maxfev = 50000, method='trf'):
        
        ## Initialize browser object
        self.spectrum1 = np.mean(self.si,axis=(0,1))
        self.spectrum2 = np.mean(self.si,axis=(0,1))

        self.im_inel = np.mean(self.si,axis=(-1))

        self.bsub1 = self.spectrum1
        self.bsub1_fit  = np.zeros_like( self.spectrum1 )
        self.bsub2 = self.spectrum2
        self.bsub2_fit  = np.zeros_like( self.spectrum2 )

        self.fit_check = False
        self.int_check = False
        self.slider_window = [0,self.ne]

        self.lc = lc
        self.gfwhm = gfwhm
        self.log = log
        self.ftol = ftol
        self.gtol = gtol
        self.xtol = xtol
        self.maxfev = maxfev
        self.method = method

        self.si_bsub = np.copy(self.si)

        self.fitfunction = 'pl'
        self.r1 = -1
        
        results_dict={}
        for key in ['bsub_spectrum','image','bsub_SI','edge']:
            results_dict[key]=[]

        
        ##############Set Initial plot#################
        self.fig=plt.figure(figsize=figsize,layout='constrained')

        self.ax = {'inel':None, 'spec':None, 'e_view':None, 'e_bsub':None, 'e_int':None,
                    'btn_fit':None, 'btn_fint':None, 'btn_int':None, 'btn_save':None}
        self.ax['inel']=self.fig.add_axes([0.025,0.1,0.45,0.8]) # Image
        self.ax['spec']=self.fig.add_axes([0.525,0.45,0.45,0.45]) # Spectrum
        self.ax['spec2'] = self.fig.add_axes([0.525,0.45,0.45,0.20]) # Spec 2
        self.ax['ck_ylock'] = self.fig.add_axes([0.85,0.35,0.13,0.05]) # Y-lock chkbox
        self.ax['ck_roi2'] = self.fig.add_axes([0.025,0.10,0.15,0.05]) # ROI2 chkbox
        self.ax['e_view']=self.fig.add_axes([0.625,0.30,0.25,0.05]) # Range slider
        self.ax['e_bsub']=self.fig.add_axes([0.625,0.25,0.25,0.05]) # Range slider
        self.ax['e_int'] =self.fig.add_axes([0.625,0.20,0.25,0.05]) # Range slider
        self.ax['btn_fit']=self.fig.add_axes([0.520,0.1,0.125,0.1]) # Fit Buttons
        self.ax['btn_fint']=self.fig.add_axes([0.655,0.1,0.1,0.1]) # Fast Int Button
        self.ax['btn_int']=self.fig.add_axes([0.765,0.1,0.1,0.1]) # Int Button 
        self.ax['btn_save']=self.fig.add_axes([0.875,0.1,0.1,0.1]) # Save Button 

        ## Initialize plot handles
        self.h = {'inel':None, 'spec':None, 'bsub1':None, 'fit':None, 'bsub2':None}
        ################## ax['inel'] ######################
        self.h['inel'] = self.ax['inel'].imshow( self.im_inel,cmap = cmap)
        self.ax['inel'].set_axis_off()
        self.ax['inel'].set_title('Inelastic image')

        ################## ax['spec'] #######################
        self.h['spec1'], =self.ax['spec'].plot(self.energy, self.spectrum1,color='crimson')
        self.h['bsub1'], =self.ax['spec'].plot(self.energy, self.bsub1,color='k',alpha=0)
        self.h['fit1'],  =self.ax['spec'].plot(self.energy, self.bsub1_fit,color='palevioletred',alpha=0)
        self.ax['spec'].axhline(0,color='k',linestyle='--',alpha=0.3)
        self.ax['spec'].set_ylim([self.spectrum1.min(),self.spectrum1.max()])
        self.ax['spec'].set_xlim([self.energy.min(),self.energy.max()])
        self.ax['spec'].set_yticks([])
        self.ax['spec'].set_xlabel('Energy (eV)')
        self.ax['spec'].set_ylabel('Intensity')
        self.ax['spec'].set_title('EELS spectrum')

        ################## ax['spec2'] #######################
        self.h['spec2'], =self.ax['spec2'].plot(self.energy, self.spectrum2,color='royalblue',alpha=0)
        self.h['bsub2'], =self.ax['spec2'].plot(self.energy, self.bsub1,color='k',alpha=0)
        self.h['fit2'],  =self.ax['spec2'].plot(self.energy, self.bsub1_fit,color='palevioletred',alpha=0)
        self.ax['spec2'].axhline(0,color='k',linestyle='--',alpha=0.3)
        self.ax['spec2'].set_ylim([self.spectrum1.min(),self.spectrum1.max()])
        self.ax['spec2'].set_xlim([self.energy.min(),self.energy.max()])
        self.ax['spec2'].set_yticks([])
        self.ax['spec2'].set_xlabel('Energy (eV)')
        self.ax['spec2'].set_ylabel('Intensity')
        # self.ax['spec2'].set_title('EELS spectrum')
        self.ax['spec2'].set_visible(False)


        ## Initialize ui handles
        self.ui = {'roi1':None, 'roi2':None, 'spec1':None, 'spec2':None,
                   'ck_ylock':None, 'ck_roi2':None,
                   'slid_e_view':None, 'slid_e_bsub':None, 'slid_e_int':None,
                   'rad_fit':None, 'btn_fint':None, 'btn_int':None, 'btn_save':None}

        ### Check box 
        # ylim locker
        self.ui['ck_ylock'] = CheckButtons(ax=self.ax['ck_ylock'], labels= ["Lock Y-axis"],
                                            actives=[False], check_props={'facecolor': 'k'} )
        self.ui['ck_ylock'].on_clicked( lambda v: self.onclick_ck_ylock() )
        self.y_locked = False

        self.ui['ck_roi2'] = CheckButtons(ax=self.ax['ck_roi2'], labels= ["Enable ROI 2"],
                                        actives=[False], check_props={'facecolor': 'k'} )
        self.ui['ck_roi2'].on_clicked( lambda v: self.onclick_ck_roi2() )
        self.roi2_enabled = False


        ################### Selectors ###################
        self.ui['roi1'] = RectangleSelector(self.ax['inel'], self.dummy, button=[1],
                                        useblit=True ,minspanx=1, minspany=1,spancoords='pixels',
                                        interactive=True,props=dict(facecolor='crimson',edgecolor='crimson',alpha=0.2,fill=True),
                                        handle_props=dict(markersize=2,markerfacecolor='white'))#,ignore_event_outside=True
        
        self.ui['roi2'] = RectangleSelector(self.ax['inel'], self.dummy, button=[3],
                                        useblit=True ,minspanx=1, minspany=1,spancoords='pixels',
                                        interactive=True,props=dict(facecolor='royalblue',edgecolor='royalblue',alpha=0.2,fill=True),
                                        handle_props=dict(markersize=2,markerfacecolor='white'))#,ignore_event_outside=True)   
        self.ui['roi2'].set_visible( False )
        self.ui['roi2'].set_active( False )
            
        self.ui['bsub'] = SpanSelector(self.ax['spec'], self.dummy, button=[1],
                                            useblit=True, minspan=1,direction="horizontal",
                                            interactive=True,props=dict(facecolor='C0',edgecolor='C0',alpha=0.2,fill=True),
                                            grab_range=10, drag_from_anywhere=True)
        self.ui['bsub2'] = SpanSelector(self.ax['spec2'], self.dummy, button=[1],
                                            useblit=True, minspan=1,direction="horizontal",
                                            interactive=True,props=dict(facecolor='C0',edgecolor='C0',alpha=0.2,fill=True),
                                            grab_range=10, drag_from_anywhere=True)
        
        self.ui['int'] = SpanSelector(self.ax['spec'], self.dummy, button=[3],
                                            useblit=True, minspan=1,direction="horizontal",
                                            interactive=True,props=dict(facecolor='orange',edgecolor='orange',alpha=0.2,fill=True),
                                            grab_range=10, drag_from_anywhere=True)
        self.ui['int2'] = SpanSelector(self.ax['spec2'], self.dummy, button=[3],
                                            useblit=True, minspan=1,direction="horizontal",
                                            interactive=True,props=dict(facecolor='orange',edgecolor='orange',alpha=0.2,fill=True),
                                            grab_range=10, drag_from_anywhere=True)
        

        ## Sliders
        self.ui['slid_e_view'] = RangeSlider(self.ax['e_view'],"Energy Range ",
                                        self.energy[0], self.energy[-1], valinit=[self.energy[0],self.energy[-1]],
                                        valstep=self.energy[1]-self.energy[0],dragging=True)
        self.ui['slid_e_view'].on_changed( self.slider_view_action )
        
        self.ui['slid_e_bsub'] = RangeSlider(self.ax['e_bsub'],"Background ",
                                            self.energy[0], self.energy[-1], valinit=[self.energy[0],self.energy[-1]],
                                            valstep=self.energy[1]-self.energy[0],dragging=True)
        self.ui['slid_e_bsub'].on_changed( self.slider_bsub_action )

        self.ui['slid_e_int'] = RangeSlider(self.ax['e_int'],"Integration ",
                                          self.energy[0],self.energy[-1],valinit=[self.energy[0],self.energy[-1]],
                                          valstep=self.energy[1]-self.energy[0],dragging=True)
        self.ui['slid_e_int'].on_changed( self.slider_int_action )


        ## Buttons
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


        self.fig.canvas.mpl_connect( 'motion_notify_event', 
                                    lambda event: self.onclick_figure(event))
        # self.ui['btn_save'].on_clicked(self.add_save_dict)
        
        # selector_collection = (spectrum_span_selector1,spectrum_span_selector2,rect_selector,ibutton,fibutton,Eslider,save_button,fitradio)
       
        if edge is None:
            self.edge = eels_edge( " ", (self.energy[0],self.energy[-1]), (self.energy[0],self.energy[-1]) )
            self.ax['e_bsub'].set_visible(False)
            self.ax['e_int'].set_visible(False)
        else:
            self.edge = edge
            if self.edge.e_bsub is None:
                self.edge.e_bsub = (self.energy[0],self.energy[-1])
                self.ax['e_bsub'].set_visible(False)
            else:
                self.fit_check = True
                self.ui['slid_e_bsub'].set_val( self.edge.e_bsub )

            if self.edge.e_int is None:
                self.edge.e_int = (self.energy[0],self.energy[-1])
                self.ax['e_int'].set_visible(False)
            else:
                self.ui['slid_e_int'].set_val( self.edge.e_int )

        self.rescale_yrange()
        # return results_dict,selector_collection
        

    ################### Update Functions ###################
    def onclick_ck_ylock(self):
        self.y_locked = self.ui['ck_ylock'].get_status()[0]
        self.rescale_yrange()

    def onclick_ck_roi2( self ):
        self.roi2_enabled = self.ui['ck_roi2'].get_status()[0]
        if self.roi2_enabled:
            self.h['spec2'].set_alpha(1)
            self.ui['roi2'].set_visible( True )
            self.ui['roi2'].set_active( True )
            self.ax['spec2'].set_visible( True )
            self.ax['spec'].set_position( [0.525,0.7,0.45,0.2] )
            if self.fit_check:
                self.update_fit2()
        else:
            self.ax['spec'].set_position( [0.525,0.45,0.45,0.45] )
            self.ax['spec2'].set_visible( False )
            self.h['spec2'].set_alpha(0)
            self.h['bsub2'].set_alpha(0)
            self.h['fit2'].set_alpha(0)
            self.ui['roi2'].set_visible( False )
            self.ui['roi2'].set_active( False )

    def update_spectrum1(self):
        real_roi = self.ui['roi1'].extents
        xmin = int( real_roi[0])
        xmax = int( real_roi[1])
        ymin = int( real_roi[2])
        ymax = int( real_roi[3])

        if xmin == xmax:
            xmax += 1
        if ymin == ymax:
            ymax +=1

        self.spectrum1=np.mean( self.si[ymin:ymax,xmin:xmax,:],axis=(0,1))

        self.h['spec1'].set_ydata( self.spectrum1)
        self.rescale_yrange()

    def update_spectrum2(self):
        real_roi = self.ui['roi2'].extents
        xmin = int( real_roi[0])
        xmax = int( real_roi[1])
        ymin = int( real_roi[2])
        ymax = int( real_roi[3])

        if xmin == xmax:
            xmax += 1
        if ymin == ymax:
            ymax +=1

        self.spectrum2=np.mean( self.si[ymin:ymax,xmin:xmax,:],axis=(0,1))

        self.h['spec2'].set_ydata( self.spectrum2)
        self.h['spec2'].set_alpha(1)
        self.rescale_yrange()
        
    def update_fit1(self):
        ind_min = np.searchsorted( self.energy, self.edge.e_bsub[0])

        self.h['bsub1'].set_ydata(self.bsub1)
        self.h['bsub1'].set_color('orangered')
        self.h['bsub1'].set_alpha(1)

        self.h['fit1'].set_data( self.energy[ind_min:], self.spectrum1[ind_min:]-self.bsub1[ind_min:])
        self.h['fit1'].set_color('palevioletred')
        self.h['fit1'].set_alpha(1)
        self.rescale_yrange()
        
    def update_fit2(self):
        ind_min = np.searchsorted( self.energy, self.edge.e_bsub[0])

        self.h['bsub2'].set_ydata(self.bsub2)
        self.h['bsub2'].set_color('steelblue')
        self.h['bsub2'].set_alpha(1)

        self.h['fit2'].set_data( self.energy[ind_min:], self.spectrum2[ind_min:]-self.bsub2[ind_min:])
        self.h['fit2'].set_color('cornflowerblue')
        self.h['fit2'].set_alpha(1)
        self.rescale_yrange()

    def update_image(self):
        self.si_bsub = self.bgsub_SI_fast( self.si, self.energy, self.edge, self.r1, fit=self.fitfunction )
        ind_min, ind_max = np.searchsorted(self.energy, self.edge.e_int)
        self.im_inel = np.mean( self.si_bsub[:,:,ind_min:ind_max],axis=(-1))
        self.h['inel'].set_array( self.im_inel)
        self.h['inel'].autoscale()
        
    def update_image_2(self):
        if self.lc:
            _,self.si_bsub = self.bgsub_SI( self.si, self.energy, self.edge, gfwhm=self.gfwhm,fit=self.fitfunction,lc=self.lc,log=self.log,
                                    ftol=self.ftol,gtol=self.gtol,xtol=self.xtol,maxfev=self.maxfev,method=self.method)
        else:
            self.si_bsub =   self.bgsub_SI( self.si, self.energy, self.edge, gfwhm=self.gfwhm,fit=self.fitfunction,lc=self.lc,log=self.log,
                                    ftol=self.ftol,gtol=self.gtol,xtol=self.xtol,maxfev=self.maxfev,method=self.method)
        indmin, indmax = np.searchsorted(self.energy, self.edge.e_int)
        self.im_inel = np.mean(self.si_bsub[:,:,indmin:indmax],axis=(-1))
        self.h['inel'].set_array(self.im_inel)
        self.h['inel'].autoscale()

    def slider_bsub_action(self, erange):
        self.ui['bsub'].extents = erange
        self.ui['bsub2'].extents = erange
        self.edge.e_bsub = erange

        self.bsub1, fit_param1 = self.bgsub_SI_linearized( self.spectrum1, self.energy, self.edge, fit=self.fitfunction)
        self.r1 = fit_param1[1]
        self.update_fit1()

        if self.roi2_enabled:
            self.bsub2, fit_param2 = self.bgsub_SI_linearized( self.spectrum2, self.energy, self.edge, fit=self.fitfunction)
            self.r2 = fit_param2[1]
            self.update_fit2()

    def slider_int_action(self, erange):
        self.ui['int'].extents = erange
        self.ui['int2'].extents = erange
        self.edge.e_int = erange

    def slider_view_action(self, erange):
        self.ax['spec'].set_xlim([erange[0],erange[1]])
        self.ax['spec2'].set_xlim([erange[0],erange[1]])

        slidermin, slidermax = np.searchsorted(self.energy, (erange[0],erange[1]))
        self.slider_window = (slidermin, slidermax)
        self.rescale_yrange()

    def rescale_yrange(self):
        if self.y_locked == False:
            slidermin,slidermax = self.slider_window

            # axis 1
            if self.fit_check:
                minval = min( 1.1*self.bsub1[slidermin:slidermax].min(),
                            -1.1*self.bsub1[slidermin:slidermax].min(),
                            0)
                
                maxval = 1.1*self.spectrum1[slidermin:slidermax].max()
            else:
                minval = min( self.spectrum1[slidermin:slidermax].min(),0 )
                maxval = 1.1*self.spectrum1[slidermin:slidermax].max()
            self.ax['spec'].set_ylim([minval,maxval])

            # axis 1
            if self.fit_check:
                minval = min( 1.1*self.bsub2[slidermin:slidermax].min(),
                            -1.1*self.bsub2[slidermin:slidermax].min(),
                            0)
                
                maxval = 1.1*self.spectrum2[slidermin:slidermax].max()
            else:
                minval = min( self.spectrum2[slidermin:slidermax].min(),0 )
                maxval = 1.1*self.spectrum2[slidermin:slidermax].max()
            self.ax['spec2'].set_ylim([minval,maxval])

    ############### Event Handlers ###################
    def onclick_figure( self, event ):
        if event.inaxes in [self.ax['inel']]:
            if event.button == MouseButton.LEFT:
                # Left Click on Inelastic Image
                self.update_spectrum1()
                
                if self.fit_check:
                    self.bsub1, fit_param = self.bgsub_SI_linearized( self.spectrum1, self.energy, self.edge, fit=self.fitfunction)
                    self.r1 = fit_param[1]
                    self.update_fit1()
            elif event.button == MouseButton.RIGHT:
                if self.roi2_enabled:
                    # Right Click on Inelastic Image
                    self.update_spectrum2()
                    if self.fit_check:
                        self.bsub2, fit_param = self.bgsub_SI_linearized( self.spectrum2, self.energy, self.edge, fit=self.fitfunction)
                        self.r2 = fit_param[1]
                        self.update_fit2()

        elif event.inaxes in [self.ax['spec']]:
            if event.button == MouseButton.LEFT:
                self.fit_check = True
                self.ax['e_bsub'].set_visible(True)
                self.ui['slid_e_bsub'].set_val( self.ui['bsub'].extents )

            elif event.button == MouseButton.RIGHT:
                self.int_check = True
                self.ax['e_int'].set_visible(True)
                self.ui['slid_e_int'].set_val( self.ui['int'].extents )
        
        elif event.inaxes in [self.ax['spec2']]:    
            if event.button == MouseButton.LEFT:
                self.fit_check = True
                self.ax['e_bsub'].set_visible(True)
                self.ui['slid_e_bsub'].set_val( self.ui['bsub2'].extents )

            elif event.button == MouseButton.RIGHT:
                self.int_check = True
                self.ax['e_int'].set_visible(True)
                self.ui['slid_e_int'].set_val( self.ui['int2'].extents )


    def fitcheck(self, label):
        fitdict = {'Power law': 'pl', 'Exponential': 'exp', 'Linear': 'lin'}
        self.fitfunction = fitdict[label]
        if self.fit_check:
            self.bsub1, fit_param = self.bgsub_SI_linearized( self.spectrum1, self.energy, self.edge, fit=self.fitfunction)
            self.r1 = fit_param[1]
            self.update_fit1()
            if self.roi2_enabled:
                self.bsub2, fit_param = self.bgsub_SI_linearized( self.spectrum2, self.energy, self.edge, fit=self.fitfunction)
                self.r2 = fit_param[1]
                self.update_fit2()
        
    def fint_button(self, event):
        if (self.int_check & self.fit_check):
            self.update_image()

    def int_button(self, event):
        if (self.int_check & self.fit_check):
            self.update_image_2()

    def dummy(self, *args):
        pass

    # def add_save_dict(event):
    #     results_dict['bsub_spectrum'] = bsub
    #     results_dict['image'] = inel_im
    #     results_dict['bsub_SI'] = bsub_array
    #     results_dict['edge'] = [fit_window[0],fit_window[1],int_window[0],int_window[1]]
    
    ### Background Subtractions
    def bgsub_SI( self, si, energy, edge, log=False, fit='pl', gfwhm=None, lc=False,
                maxfev=50000, method='trf', ftol=0.0005, gtol=0.00005, xtol=None, **kwargs):
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

        fit_start_ch, fit_end_ch = np.searchsorted( energy, edge.e_bsub)
        si = si.astype('float32')
        if len(np.shape(si)) == 2:
            tempx,tempz = np.shape(si)
            si = np.reshape(si,(tempx,1,tempz))
        if len(np.shape(si)) == 1:
            tempz = len(si)
            si = np.reshape(si,(1,1,tempz))
        xdim, ydim, zdim = np.shape(si)

        ## Special case: if there is vacuum in the SI and it is causing trouble with your LCPL fitting:
        if 'mask' in kwargs.keys():
            mask = kwargs['mask']
        elif 'threshold' in kwargs.keys():
            thresh = kwargs['threshold']
            mean_back = np.mean(si[:,:,fit_start_ch:fit_end_ch],axis=2)
            mask = mean_back > thresh
        else:
            mask = np.ones((xdim,ydim), dtype='bool')

        ## Apply Local Background Averaging
        if gfwhm is not None:
            fit_data = self.prepare_lba( si, gfwhm, fit_start_ch, fit_end_ch )
        else:
            fit_data = si
        
        ## If log fitting or linear fitting, find fit using qr factorization       
        if log | (fit=='lin'):
            bg_pl_SI, fit_params = self.bgsub_SI_linearized( fit_data, self.energy, self.edge, fit=fit )

            maskline = np.reshape( mask,(xdim*ydim))
            rline_long = -1*np.reshape( fit_params[1,:,:], (xdim*ydim) )
            rline = rline_long[maskline]

        ## Power law non-linear curve fitting using scipy.optimize.curve_fit
        elif (fit=='pl') | (fit=='exp') : 
            bg_pl_SI, fit_params = self.bgsub_SI_nllsq( fit_data, self.energy, self.edge, fit=fit, 
                                            maxfev=maxfev, method=method,
                                            ftol=ftol, gtol=gtol, xtol=xtol )

        ## Given r values of SI, refit background using a linear combination of power laws, 
        ## using either 5/95 percentile or 20/80 percentile r values.
        if lc:
            bg_lcpl_SI = self.bgsub_SI_LC(si, self.energy, self.edge, rline, fit=fit, nstd=2)
            return bg_pl_SI, bg_lcpl_SI
        else:
            return bg_pl_SI
        
    def prepare_lba( self, si, gfwhm, fit_start_ch, fit_end_ch ):
        lba_raw = np.copy( si )
        lba_normalized = np.copy( si )
        for energychannel in np.arange(fit_start_ch,fit_end_ch):
            lba_raw[:,:,energychannel] = gaussian_filter(si[:,:,energychannel],sigma=gfwhm/2.35)
        
        lba_mean = np.mean( lba_raw[:,:,fit_start_ch:fit_end_ch], 2 )
        data_mean = np.mean(     si[:,:,fit_start_ch:fit_end_ch], 2)

        for energychannel in np.arange(fit_start_ch,fit_end_ch):
            lba_normalized[:,:,energychannel] = lba_raw[:,:,energychannel]*data_mean/lba_mean

        return lba_normalized
        

    def linear_regression_QR( self, y, X ):
        # Solve Linear Regression using QR Decomposition
        # Y = Xb + error
        # R*b = Q.T*y minimizes MSE
        # y: (nx1) dependent variable
        # x: (nx1) independent variable
        # b: (2x1) [[b0],[b1]], b0: intercept, b1: slope
        if y.ndim == 1:
            Y = np.atleast_2d( y ).T
        else:
            Y = y

        Q, R = LA.qr(X)
        b = LA.inv(R) @ (Q.T @ Y)

        return b


 



    ########### background subtractions ########
    def bgsub_SI_fast( self, si, energy, edge, rval, fit='pl'):
        """
        Quick background subtraction based on fixed 'r' value
        For Y = Ax + b + error with fixed 'A':
            Y' = b + error. MSE is minimized when b = mean(Y)
        """
        xdim, ydim, zdim = np.shape( si )

        fit_start_ch, fit_end_ch = np.searchsorted(energy, edge.e_bsub)
        y_win = si[:,:,fit_start_ch:fit_end_ch]
        e_win = np.reshape( energy[fit_start_ch:fit_end_ch], (1,1,(fit_end_ch-fit_start_ch)) )
        e_sub = np.reshape( energy[fit_start_ch:], (1,1,zdim-fit_start_ch) )

        bg_SI = np.zeros_like( si )  

        if fit == 'lin':
            c_fit = np.reshape( np.mean( y_win-rval*e_win, axis=(2)), (xdim,ydim,1))
            y_fit = c_fit + rval*e_sub

        if fit == 'pl':
            c_fit = np.reshape( np.mean( np.log(y_win)-rval*np.log(e_win), axis=(2)), (xdim,ydim,1))
            y_fit = np.exp( c_fit + rval*np.log(e_sub) )

        if fit == 'exp':
            c_fit = np.reshape( np.mean( np.log(y_win)-rval*e_win, axis=(2)), (xdim,ydim,1))
            y_fit = np.exp( c_fit + rval*e_sub )

        bg_SI[:,:,fit_start_ch:] = si[:,:,fit_start_ch:] - y_fit
        return bg_SI

    def bgsub_SI_linearized( self, si, energy, edge, fit='pl'):
        """
        Quick background subtraction based on fixed 'r' value
        For Y = Ax + b + error with fixed 'A':
            Y' = b + error. MSE is minimized when b = mean(Y)
        """

        fit_start_ch, fit_end_ch = np.searchsorted(energy, edge.e_bsub)
        if (fit_end_ch - fit_start_ch)<2:
            fit_end_ch = fit_start_ch+2
        e_win = np.atleast_2d( energy[fit_start_ch:fit_end_ch] ).T
        e_sub = np.atleast_2d( energy[fit_start_ch:] ).T
        zdim = len(energy)

        if si.ndim == 1:
            si = np.reshape( si, (1,1,zdim))
        elif si.ndim == 2:
            (nx,nz) = si.shape
            si = np.reshape( si,(nx,1,zdim))

        xdim, ydim, zdim = np.shape( si )
        y_win = si[:,:,fit_start_ch:fit_end_ch]
        y_win = np.reshape( y_win, (xdim*ydim, len(e_win))).T

        bg_SI = np.zeros_like( si )  
        if fit == 'lin':
            e_win = np.insert( e_win, 0, 1, axis=1)
            e_sub = np.insert( e_sub, 0, 1, axis=1)

            b_fit = self.linear_regression_QR( y_win, e_win)
            y_fit = e_sub @ b_fit

        if fit == 'pl':
            e_win = np.insert( np.log(e_win), 0, 1, axis=1)
            e_sub = np.insert( np.log(e_sub), 0, 1, axis=1)

            b_fit = self.linear_regression_QR( np.log(y_win), e_win )
            y_fit = np.exp(e_sub @ b_fit )
            
            b_fit[0,:] = np.exp( b_fit[0,:] )

        if fit == 'exp':
            e_win = np.insert( e_win, 0, 1, axis=1)
            e_sub = np.insert( e_sub, 0, 1, axis=1)

            b_fit = self.linear_regression_QR( np.log(y_win), e_win )
            y_fit = np.exp(e_sub @ b_fit )

            b_fit[0,:] = np.exp( b_fit[0,:] )

        b_fit = np.squeeze( np.reshape( b_fit, (2,xdim,ydim)) )
        y_fit = np.reshape( y_fit.T, (xdim,ydim,len(e_sub)))
        bg_SI[:,:,fit_start_ch:] = si[:,:,fit_start_ch:] - y_fit
        bg_SI = np.squeeze( bg_SI )
        return bg_SI, b_fit

    def bgsub_SI_nllsq( self, si, energy, edge, fit='pl',gfwhm=None, 
                    maxfev=50000, method='trf', ftol=0.0005, gtol=0.00005, xtol=None):
        """
        Quick background subtraction based on fixed 'r' value
        For Y = Ax + b + error with fixed 'A':
            Y' = b + error. MSE is minimized when b = mean(Y)
        """

        fit_start_ch, fit_end_ch = np.searchsorted(energy, edge.e_bsub)
        e_win = energy[fit_start_ch:fit_end_ch]
        e_sub = energy[fit_start_ch:]
        zdim = len(energy)

        if si.ndim == 1:
            si = np.reshape( si, (1,1,zdim))
        elif si.ndim == 2:
            (nx,nz) = si.shape
            si = np.reshape( si,(nx,1,zdim))

        xdim, ydim, zdim = np.shape( si )
        y_win = si[:,:,fit_start_ch:fit_end_ch]
        bg_SI = np.zeros_like( si )

        if fit == 'pl':
            fitfunc = ls.powerlaw
            fit_params = np.zeros(2,xdim,ydim)
        elif fit == 'exp':
            fitfunc = ls.exponential
            fit_params = np.zeros(2,xdim,ydim)

        mean_spec = np.mean( y_win, (0,1) )
        popt_init,pcov_init = curve_fit( fitfunc, e_win, mean_spec, maxfev=maxfev,method=method,verbose=0 )
        
        pbar1 = tqdm(total = (xdim)*(ydim),desc = "Background subtracting")
        for i in range(xdim):
            for j in range(ydim):
                popt_pl,pcov_pl=curve_fit( fitfunc, e_win, y_win[i,j,:],p0=popt_init,
                                        maxfev=maxfev,method=method,verbose = 0,
                                        ftol=ftol, gtol=gtol, xtol=xtol)
                
                bg_SI[i,j,fit_start_ch:] = si[i,j,fit_start_ch:] - fitfunc(energy[fit_start_ch:], *popt_pl)
                fit_params[:,i,j] = popt_pl
                pbar1.update(1)

        return bg_SI, fit_params

    def bgsub_SI_LC( self, si, energy, edge, rline, fit='pl', nstd=2):
        bg_lcpl_SI = np.zeros_like(si)
        rmu,rstd = norm.fit(rline)
        rmin = rmu - nstd*rstd
        rmax = rmu + nstd*rstd

        (xdim, ydim, zdim) = si.shape
        fit_start_ch, fit_end_ch = np.searchsorted(energy, edge.e_bsub)

        
        if fit=='pl':
            fitname = 'power law'
            e_win = np.log(energy[fit_start_ch:fit_end_ch])
            e_sub = np.log(energy[fit_start_ch:])
        elif fit=='exp':
            fitname = 'exponential'
            e_win = energy[fit_start_ch:fit_end_ch]
            e_sub = energy[fit_start_ch:]

        if nstd == 2:
            print( '5th percentile {} = {}'.format( fitname, rmin))
            print('95th percentile {} = {}'.format( fitname, rmax))
        elif nstd == 1:
            print('20th percentile {} = {}'.format( fitname, rmin))
            print('80th percentile {} = {}'.format( fitname, rmax))
        else:
            print('Min {} = {}'.format( fitname, rmin))
            print('Max {} = {}'.format( fitname, rmax))


        e_win = np.atleast_2d( energy[fit_start_ch:fit_end_ch] ).T
        e_win = np.append( e_win**(-rmin), e_win**(-rmax), axis=1)
        e_sub = np.atleast_2d( energy[fit_start_ch:] )
        e_sub = np.append( e_sub**(-rmin), e_sub**(-rmax), axis=1)
        
        y_win = np.reshape( si[:,:,fit_start_ch:fit_end_ch], (xdim*ydim,len(e_win) ) )
        b_fit = self.linear_regression_QR( y_win, e_win )
        y_fit = e_sub @ b_fit

        bgndLCPL = np.reshape( y_fit,(xdim,ydim,len(e_sub)))
        bg_lcpl_SI[:,:,fit_start_ch:] = si[:,:,fit_start_ch:] - bgndLCPL

        return bg_lcpl_SI

