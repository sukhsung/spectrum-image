import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, RectangleSelector, SpanSelector, CheckButtons, RadioButtons, Button, TextBox
from matplotlib.backend_bases import MouseButton

import spectrum_image.EELS.EELS_bgsub as bg
from spectrum_image.EELS.EELS_edge import EELS_edge

class SpectrumImage :
    def __init__( self, si, energy, adf=None, xaxis=None, yaxis=None ):

        si[np.isnan(si)] = 0
        self.si = si
        (self.ny, self.nx, self.ne) = self.si.shape
        self.eaxis = np.asarray( energy )

        self.adf = adf
        if xaxis is None:
            xaxis = np.arange( self.nx )
        if yaxis is None:
            yaxis = np.arange( self.ny )

        self.xaxis = xaxis
        self.yaxis = yaxis
        
    def fitbrowser( self, edge=None, cmap='gray', figsize=(9,6)):
        
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

        self.fit_options = bg.options_bgsub()
        self.fit_options.lc = False
        self.fit_options.lba = False
        self.fit_options.log = True
        self.fit_options.gfwhm = 5

        self.si_bsub = None

        self.r1 = -1
                
        ##############Set Initial plot#################
        self.fig=plt.figure(figsize=figsize,layout='constrained')

        self.ax = {}
        self.ax['inel']=self.fig.add_axes([0.05,0.1,0.43,0.8]) # Image
        self.ax['spec']=self.fig.add_axes([0.525,0.45,0.45,0.45]) # Spectrum
        self.ax['spec2'] = self.fig.add_axes([0.525,0.45,0.45,0.20]) # Spec 2
        self.ax['ck_ysetting'] = self.fig.add_axes([0.85,0.89,0.13,0.07]) # Y-lock chkbox
        self.ax['ck_roi2'] = self.fig.add_axes([0.025,0.0,0.15,0.05]) # ROI2 chkbox
        if self.adf is not None:
            self.ax['ck_adf'] = self.fig.add_axes([0.18,0.0,0.15,0.05]) # adf chkbox
        self.ax['e_view']=self.fig.add_axes( [0.625,0.30,0.25,0.05]) # Range slider
        self.ax['e_bsub']=self.fig.add_axes([0.625,0.25,0.25,0.05]) # Range slider
        self.ax['e_int'] =self.fig.add_axes([0.625,0.20,0.25,0.05]) # Range slider
        self.ax['btn_fit']=self.fig.add_axes([0.520,0.1,0.125,0.1]) # Fit Buttons
        self.ax['btn_fbsub']=self.fig.add_axes([0.655,0.1,0.1,0.1]) # Fast Int Button
        self.ax['btn_bsub']=self.fig.add_axes([0.765,0.1,0.1,0.1]) # Int Button 
        self.ax['ck_fit']=self.fig.add_axes([0.875,0.1,0.1,0.1]) # Axis for Fit settings

        self.ax['lc_tx'] = self.fig.add_axes([0.875,0.06,0.1,0.025]) # LC settings
        self.ax['lba_tx'] = self.fig.add_axes([0.875,0.03,0.1,0.025]) # LBA settings

        ## Initialize plot handles
        self.h = {}
        ################## ax['inel'] ######################
        # self.h['inel'] = self.ax['inel'].matshow( self.im_inel,cmap = cmap)
        self.h['inel'] = self.ax['inel'].pcolormesh( self.xaxis, self.yaxis, self.im_inel,cmap = cmap)
        self.ax['inel'].set_axis_on()
        self.ax['inel'].set_title('Inelastic image')

        ################## ax['spec'] #######################
        self.h['spec1'], =self.ax['spec'].plot(self.eaxis, self.spectrum1,color='crimson')
        self.h['bsub1'], =self.ax['spec'].plot(self.eaxis, self.bsub1,color='k',alpha=0)
        self.h['fit1'],  =self.ax['spec'].plot(self.eaxis, self.bsub1_fit,color='palevioletred',alpha=0)
        self.ax['spec'].axhline(0,color='k',linestyle='--',alpha=0.3)
        self.ax['spec'].set_ylim([self.spectrum1.min(),self.spectrum1.max()])
        self.ax['spec'].set_xlim([self.eaxis.min(),self.eaxis.max()])
        self.ax['spec'].set_yticks([])
        self.ax['spec'].set_xlabel('Energy (eV)')
        self.ax['spec'].set_ylabel('Intensity')
        self.ax['spec'].set_title('EELS spectrum')

        ################## ax['spec2'] #######################
        self.h['spec2'], =self.ax['spec2'].plot(self.eaxis, self.spectrum2,color='royalblue',alpha=0)
        self.h['bsub2'], =self.ax['spec2'].plot(self.eaxis, self.bsub1,color='k',alpha=0)
        self.h['fit2'],  =self.ax['spec2'].plot(self.eaxis, self.bsub1_fit,color='palevioletred',alpha=0)
        self.ax['spec2'].axhline(0,color='k',linestyle='--',alpha=0.3)
        self.ax['spec2'].set_ylim([self.spectrum1.min(),self.spectrum1.max()])
        self.ax['spec2'].set_xlim([self.eaxis.min(),self.eaxis.max()])
        self.ax['spec2'].set_yticks([])
        self.ax['spec2'].set_xlabel('Energy (eV)')
        self.ax['spec2'].set_ylabel('Intensity')
        self.ax['spec2'].set_visible(False)

        ## Initialize ui handles
        self.ui={}
        ### Check box 
        # ylim locker
        self.ui['ck_ysetting'] = CheckButtons(ax=self.ax['ck_ysetting'], labels= ["Lock Y-axis","Log(y)"],
                                            actives=[False, False], check_props={'facecolor': 'k'} )
        self.ui['ck_ysetting'].on_clicked( lambda v: self.onclick_ck_ysetting() )
        self.y_locked = False
        self.y_log = False

        self.ui['ck_fit'] = CheckButtons(ax=self.ax['ck_fit'], labels= ["LC", "LBA", "log"],
                                            actives=[False, False, True], check_props={'facecolor': 'k'} )
        self.ui['ck_fit'].on_clicked( lambda v: self.onclick_ck_fit() )

        self.ui['ck_roi2'] = CheckButtons(ax=self.ax['ck_roi2'], labels= ["Enable ROI 2"],
                                        actives=[False], check_props={'facecolor': 'k'} )
        self.ui['ck_roi2'].on_clicked( lambda v: self.onclick_ck_roi2() )
        self.roi2_enabled = False

        if self.adf is not None:
            self.ui['ck_adf'] = CheckButtons(ax=self.ax['ck_adf'], labels= ["Toggle ADF"],
                                            actives=[False], check_props={'facecolor': 'k'} )
            self.ui['ck_adf'].on_clicked( lambda v: self.onclick_ck_adf() )
        self.adf_enabled = False


        self.ui['lc_tx'] = TextBox(self.ax['lc_tx'], "LC: ", textalignment="left")
        self.ui['lc_tx'].on_submit( self.onchange_lc )
        self.ui['lc_tx'].set_val( '(5, 95)')
        self.ui['lc_tx'].set_active( False )
        self.ui['lc_tx'].text_disp.set_color((0.75, 0.75, 0.75))
        self.ui['lba_tx'] = TextBox(self.ax['lba_tx'], "LBA: ", textalignment="left")
        self.ui['lba_tx'].on_submit( self.onchange_lba )
        self.ui['lba_tx'].set_val( '5')
        self.ui['lba_tx'].set_active( False )
        self.ui['lba_tx'].text_disp.set_color((0.75, 0.75, 0.75))



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
                                        self.eaxis[0], self.eaxis[-1], valinit=[self.eaxis[0],self.eaxis[-1]],
                                        valstep=self.eaxis[1]-self.eaxis[0],dragging=True)
        self.ui['slid_e_view'].on_changed( self.slider_view_action )
        
        self.ui['slid_e_bsub'] = RangeSlider(self.ax['e_bsub'],"Background ",
                                            self.eaxis[0], self.eaxis[-1], valinit=[self.eaxis[0],self.eaxis[-1]],
                                            valstep=self.eaxis[1]-self.eaxis[0],dragging=True)
        self.ui['slid_e_bsub'].on_changed( self.slider_bsub_action )

        self.ui['slid_e_int'] = RangeSlider(self.ax['e_int'],"Integration ",
                                          self.eaxis[0],self.eaxis[-1],valinit=[self.eaxis[0],self.eaxis[-1]],
                                          valstep=self.eaxis[1]-self.eaxis[0],dragging=True)
        self.ui['slid_e_int'].on_changed( self.slider_int_action )


        ## Buttons
        self.ax['btn_fit'].set_facecolor('0.85')
        self.ui['rad_fit'] = RadioButtons(self.ax['btn_fit'], ('Power law', 'Exponential', 'Linear'),
                            label_props={'color': ['k','k','k'], 'fontsize': [10, 10, 10]},
                            radio_props={'s': [16,16,16]})
        
        self.ui['rad_fit'].on_clicked(self.onclick_fitmode)

        self.ui['btn_fbsub']=Button(self.ax['btn_fbsub'],"Fast\nSubtraction",useblit=True,)
        self.ui['btn_fbsub'].on_clicked( lambda v: self.onclick_fbsub() )

        self.ui['btn_bsub']=Button(self.ax['btn_bsub'],"Background\nSubtraction",useblit=True,)
        self.ui['btn_bsub'].on_clicked( lambda v: self.onclick_bsub() )


        self.fig.canvas.mpl_connect( 'motion_notify_event', 
                                    lambda event: self.onclick_figure(event))
        
        if edge is None:
            self.edge = EELS_edge( " ", (self.eaxis[0],self.eaxis[-1]), (self.eaxis[0],self.eaxis[-1]) )
            self.ax['e_bsub'].set_visible(False)
            self.ax['e_int'].set_visible(False)
        else:
            self.edge = edge
            if self.edge.e_bsub is None:
                self.edge.e_bsub = (self.eaxis[0],self.eaxis[-1])
                self.ax['e_bsub'].set_visible(False)
            else:
                self.fit_check = True
                self.ui['slid_e_bsub'].set_val( self.edge.e_bsub )

            if self.edge.e_int is None:
                self.edge.e_int = (self.eaxis[0],self.eaxis[-1])
                self.ax['e_int'].set_visible(False)
            else:
                self.int_check = True
                self.ui['slid_e_int'].set_val( self.edge.e_int )

        self.rescale_yrange()
        # return results_dict,selector_collection
        
    ################### Update Functions ###################
    def onchange_lc(self, value ):
        self.fit_options.perc = eval( value )

    def onchange_lba(self, value ):
        self.fit_options.gfwhm = eval( value )

    def onclick_ck_adf(self) :
        self.adf_enabled = self.ui['ck_adf'].get_status()[0]
        if self.adf_enabled:
            self.h['inel'].set_array(self.adf)
            self.h['inel'].autoscale()
            self.ax['inel'].set_title('ADF Image')
        else:
            self.h['inel'].set_array(self.im_inel)
            self.h['inel'].autoscale()
            self.ax['inel'].set_title('Inelastic Image')

    def onclick_ck_fit( self ):
        self.fit_options.lc = self.ui['ck_fit'].get_status()[0]
        self.fit_options.lba = self.ui['ck_fit'].get_status()[1]
        self.fit_options.log = self.ui['ck_fit'].get_status()[2]

        if self.fit_options.lc == True and self.fit_options.fit == 'lin':
            self.fit_options.lc = False
            self.ui['ck_fit'].set_active(0)

        self.ui['lba_tx'].set_active( self.fit_options.lba )
        self.ui['lc_tx'].set_active( self.fit_options.lc )

        if self.fit_options.lba:
            self.ui['lba_tx'].text_disp.set_color((0,0,0))
        else:
            self.ui['lba_tx'].text_disp.set_color((0.75, 0.75, 0.75))

        if self.fit_options.lc:
            self.ui['lc_tx'].text_disp.set_color((0,0,0))
        else:
            self.ui['lc_tx'].text_disp.set_color((0.75, 0.75, 0.75))

    
    def onclick_ck_ysetting(self):
        self.y_locked = self.ui['ck_ysetting'].get_status()[0]
        self.y_log = self.ui['ck_ysetting'].get_status()[1]
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
                self.calc_bsub2()
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
        xmin = np.searchsorted( self.xaxis, int( real_roi[0]))
        xmax = np.searchsorted( self.xaxis, int( real_roi[1]))
        ymin = np.searchsorted( self.yaxis, int( real_roi[2]))
        ymax = np.searchsorted( self.yaxis, int( real_roi[3]))

        if xmin == xmax:
            xmax += 1
        if ymin == ymax:
            ymax +=1

        self.spectrum1=np.mean( self.si[ymin:ymax,xmin:xmax,:],axis=(0,1))
        
        self.h['spec1'].set_ydata( self.spectrum1)
        self.rescale_yrange()

    def update_spectrum2(self):
        real_roi = self.ui['roi2'].extents
        xmin = np.searchsorted( self.xaxis, int( real_roi[0]))
        xmax = np.searchsorted( self.xaxis, int( real_roi[1]))
        ymin = np.searchsorted( self.yaxis, int( real_roi[2]))
        ymax = np.searchsorted( self.yaxis, int( real_roi[3]))

        if xmin == xmax:
            xmax += 1
        if ymin == ymax:
            ymax +=1

        self.spectrum2=np.mean( self.si[ymin:ymax,xmin:xmax,:],axis=(0,1))

        self.h['spec2'].set_ydata( self.spectrum2)
        self.h['spec2'].set_alpha(1)
        self.rescale_yrange()
        
    def update_fit1(self):
        ind_min = np.searchsorted( self.eaxis, self.edge.e_bsub[0])

        self.h['bsub1'].set_ydata(self.bsub1)
        self.h['bsub1'].set_color('orangered')
        self.h['bsub1'].set_alpha(1)

        self.h['fit1'].set_data( self.eaxis[ind_min:], self.spectrum1[ind_min:]-self.bsub1[ind_min:])
        self.h['fit1'].set_color('palevioletred')
        self.h['fit1'].set_alpha(1)
        self.rescale_yrange()
        
    def update_fit2(self):
        ind_min = np.searchsorted( self.eaxis, self.edge.e_bsub[0])

        self.h['bsub2'].set_ydata(self.bsub2)
        self.h['bsub2'].set_color('steelblue')
        self.h['bsub2'].set_alpha(1)

        self.h['fit2'].set_data( self.eaxis[ind_min:], self.spectrum2[ind_min:]-self.bsub2[ind_min:])
        self.h['fit2'].set_color('cornflowerblue')
        self.h['fit2'].set_alpha(1)
        self.rescale_yrange()

    def slider_bsub_action(self, erange):
        self.ui['bsub'].extents = erange
        self.ui['bsub2'].extents = erange
        self.edge.e_bsub = erange

        if self.fit_check:
            self.calc_bsub1()
            self.update_fit1()
            self.calc_bsub2()
            self.update_fit2()

    def slider_int_action(self, erange):
        self.ui['int'].extents = erange
        self.ui['int2'].extents = erange
        self.edge.e_int = erange

        if self.adf_enabled:
            self.ui['ck_adf'].set_active(0)
        self.update_image()

    def slider_view_action(self, erange):
        self.ax['spec'].set_xlim([erange[0],erange[1]])
        self.ax['spec2'].set_xlim([erange[0],erange[1]])

        slidermin, slidermax = np.searchsorted(self.eaxis, (erange[0],erange[1]))
        self.slider_window = (slidermin, slidermax)
        self.rescale_yrange()

    def rescale_yrange(self):

        if self.y_log:
            self.ax['spec'].set_yscale('log')
            self.ax['spec'].set_ylabel('Log Intensity')
            self.ax['spec'].set_yticks([])
            self.ax['spec2'].set_yscale('log')
            self.ax['spec2'].set_ylabel('Log Intensity')
            self.ax['spec2'].set_yticks([])
        else:
            self.ax['spec'].set_yscale('linear')
            self.ax['spec'].set_ylabel('Intensity')
            self.ax['spec'].set_yticks([])
            self.ax['spec2'].set_yscale('linear')
            self.ax['spec2'].set_ylabel('Intensity')
            self.ax['spec2'].set_yticks([])

        if self.y_locked == False:
            slidermin,slidermax = self.slider_window
            
            if not self.y_log:
                maxval =  1.1*self.spectrum1[slidermin:slidermax].max()
                minval =  min( 0.9*self.spectrum1[slidermin:slidermax].min(),0)

                if self.fit_check:
                    minval = min( 0.9*self.bsub1[slidermin:slidermax].min(),
                                  0)
            else:
                maxval =  1.2*self.spectrum1[slidermin:slidermax].max()
                minval =  0.8*self.spectrum1[slidermin:slidermax].min()

            self.ax['spec'].set_ylim([minval,maxval])
            self.ax['spec'].set_yticks([])

            # axis 2
            if self.roi2_enabled:
                if not self.y_log:
                    maxval =  1.1*self.spectrum2[slidermin:slidermax].max()
                    minval =  min( 0.9*self.spectrum2[slidermin:slidermax].min(),0)

                    if self.fit_check:
                        minval = min( 0.9*self.bsub2[slidermin:slidermax].min(),
                                    0)
                else:
                    maxval =  1.2*self.spectrum2[slidermin:slidermax].max()
                    minval =  0.8*self.spectrum2[slidermin:slidermax].min()

                self.ax['spec2'].set_ylim([minval,maxval])


    ############### Event Handlers ###################
    def onclick_figure( self, event ):
        if event.inaxes in [self.ax['inel']]:
            if event.button == MouseButton.LEFT:
                # Left Click on Inelastic Image
                self.update_spectrum1()
                if self.fit_check:
                    self.calc_bsub1()
                    self.update_fit1()
            elif event.button == MouseButton.RIGHT:
                if self.roi2_enabled:
                    # Right Click on Inelastic Image
                    self.update_spectrum2()
                    if self.fit_check:
                        self.calc_bsub2()
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

    def calc_bsub1(self):
            self.bsub1, fit_param = bg.bgsub_SI_linearized( self.spectrum1, self.eaxis, self.edge, fit_options=self.fit_options)
            self.r1 = fit_param[1]

    def calc_bsub2(self):
        self.bsub2, fit_param = bg.bgsub_SI_linearized( self.spectrum2, self.eaxis, self.edge, fit_options=self.fit_options)
        self.r2 = fit_param[1]


    def onclick_fitmode(self, label):
        fitdict = {'Power law': 'pl', 'Exponential': 'exp', 'Linear': 'lin'}
        self.fit_options.fit = fitdict[label]
        if self.fit_options.fit == 'lin' and self.fit_options.lc==True:
            self.ui['ck_fit'].set_active(0)

        self.calc_bsub1()
        self.update_fit1()
        self.calc_bsub2()
        self.update_fit1()
        
    def onclick_fbsub(self):
        if (self.int_check and self.fit_check):
            self.si_bsub = bg.bgsub_SI_fast( self.si, self.eaxis, self.edge, self.r1, fit_options=self.fit_options)
            
            self.update_image()


    def onclick_bsub(self):
        if (self.int_check and self.fit_check):
            if self.fit_options.lc:
                _,self.si_bsub = bg.bgsub_SI( self.si, self.eaxis, self.edge, fit_options=self.fit_options)
            else:
                self.si_bsub =   bg.bgsub_SI( self.si, self.eaxis, self.edge, fit_options=self.fit_options)
            self.update_image()

    def update_image(self):
        indmin, indmax = np.searchsorted(self.eaxis, self.edge.e_int)
        if indmin == indmax:
            indmax +1

        if self.si_bsub is None:
            self.im_inel = np.mean(self.si[:,:,indmin:indmax],axis=(-1))
        else:           
            self.im_inel = np.mean(self.si_bsub[:,:,indmin:indmax],axis=(-1))
        
        if not self.adf_enabled:
            self.h['inel'].set_array(self.im_inel)
            self.h['inel'].autoscale()


    def dummy(self, *args):
        pass
