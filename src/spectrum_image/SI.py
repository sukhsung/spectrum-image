import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, RectangleSelector, SpanSelector, CheckButtons, RadioButtons, Button, TextBox
from matplotlib.backend_bases import MouseButton
import spectrum_image.eels_bgsub as bg

class eels_edge:
    def __init__( self, label="", e_bsub=None, e_int=None ):
        self.label  = label
        self.e_bsub = e_bsub
        self.e_int  = e_int

    def from_KEM( self, edge_KEM ):
        # construct edge from KEM convention
        # edge_KEM=['label',bsub_start,bsub_end,int_start,int_end]
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
        self.fit_options.log = False
        self.fit_options.gfwhm = 5

        self.si_bsub = None

        self.r1 = -1
                
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
        self.ax['ck_fit']=self.fig.add_axes([0.875,0.1,0.1,0.1]) # Axis for Fit settings

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

        self.ui['ck_fit'] = CheckButtons(ax=self.ax['ck_fit'], labels= ["LC", "LBA", "log"],
                                            actives=[False, False, False], check_props={'facecolor': 'k'} )
        # text_box = TextBox(self.ax['btn_save'], "LBA", textalignment="left")
        self.ui['ck_fit'].on_clicked( lambda v: self.onclick_ck_fit() )

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

        self.ui['btn_fint']=Button(self.ax['btn_fint'],"Fast\nSubtraction",useblit=True,)
        self.ui['btn_fint'].on_clicked(self.fint_button)

        self.ui['btn_int']=Button(self.ax['btn_int'],"Background\nSubtraction",useblit=True,)
        self.ui['btn_int'].on_clicked(self.int_button)


        self.fig.canvas.mpl_connect( 'motion_notify_event', 
                                    lambda event: self.onclick_figure(event))
        
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
                self.int_check = True
                self.ui['slid_e_int'].set_val( self.edge.e_int )

        self.rescale_yrange()
        # return results_dict,selector_collection
        

    ################### Update Functions ###################
    def onclick_ck_fit( self ):
        self.fit_options.lc = self.ui['ck_fit'].get_status()[0]
        self.fit_options.lba = self.ui['ck_fit'].get_status()[1]
        self.fit_options.log = self.ui['ck_fit'].get_status()[2]
        if self.fit_options.lc == True and self.fit_options.fit == 'lin':
            self.fit_options.lc = False
            self.ui['ck_fit'].set_active(0)
    
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

    def slider_bsub_action(self, erange):
        self.ui['bsub'].extents = erange
        self.ui['bsub2'].extents = erange
        self.edge.e_bsub = erange

        self.bsub1, fit_param1 = bg.bgsub_SI_linearized( self.spectrum1, self.energy, self.edge, fit_options=self.fit_options)
        self.r1 = fit_param1[1]
        self.update_fit1()

        if self.roi2_enabled:
            self.bsub2, fit_param2 = bg.bgsub_SI_linearized( self.spectrum2, self.energy, self.edge, fit_options=self.fit_options)
            self.r2 = fit_param2[1]
            self.update_fit2()

    def slider_int_action(self, erange):
        self.ui['int'].extents = erange
        self.ui['int2'].extents = erange
        self.edge.e_int = erange
        self.update_image()

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
                maxval =  1.1*self.spectrum1[slidermin:slidermax].max()
            self.ax['spec'].set_ylim([minval,maxval])

            # axis 2
            if self.roi2_enabled:
                if self.fit_check:
                    minval = min( 1.1*self.bsub2[slidermin:slidermax].min(),
                                -1.1*self.bsub2[slidermin:slidermax].min(),
                                0)
                    
                    maxval = 1.1*self.spectrum2[slidermin:slidermax].max()
                else:
                    minval = min( self.spectrum2[slidermin:slidermax].min(),0 )
                    maxval =  1.1*self.spectrum2[slidermin:slidermax].max()
                self.ax['spec2'].set_ylim([minval,maxval])

    ############### Event Handlers ###################
    def onclick_figure( self, event ):
        if event.inaxes in [self.ax['inel']]:
            if event.button == MouseButton.LEFT:
                # Left Click on Inelastic Image
                self.update_spectrum1()
                
                if self.fit_check:
                    self.bsub1, fit_param = bg.bgsub_SI_linearized( self.spectrum1, self.energy, self.edge, fit_options=self.fit_options)
                    self.r1 = fit_param[1]
                    self.update_fit1()
            elif event.button == MouseButton.RIGHT:
                if self.roi2_enabled:
                    # Right Click on Inelastic Image
                    self.update_spectrum2()
                    if self.fit_check:
                        self.bsub2, fit_param = bg.bgsub_SI_linearized( self.spectrum2, self.energy, self.edge, fit_options=self.fit_options)
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
        self.fit_options.fit = fitdict[label]
        if self.fit_options.fit == 'lin' and self.fit_options.lc==True:
            self.ui['ck_fit'].set_active(0)
        if self.fit_check:
            self.bsub1, fit_param = bg.bgsub_SI_linearized( self.spectrum1, self.energy, self.edge, fit_options=self.fit_options)
            self.r1 = fit_param[1]
            self.update_fit1()
            if self.roi2_enabled:
                self.bsub2, fit_param = bg.bgsub_SI_linearized( self.spectrum2, self.energy, self.edge, fit_options=self.fit_options)
                self.r2 = fit_param[1]
                self.update_fit2()
        
    def fint_button(self, event):
        if (self.int_check and self.fit_check):
            self.si_bsub = bg.bgsub_SI_fast( self.si, self.energy, self.edge, self.r1, fit_options=self.fit_options)
            
            self.update_image()


    def int_button(self, event):
        if (self.int_check and self.fit_check):
            if self.fit_options.lc:
                _,self.si_bsub = bg.bgsub_SI( self.si, self.energy, self.edge, fit_options=self.fit_options)
            else:
                self.si_bsub =   bg.bgsub_SI( self.si, self.energy, self.edge, fit_options=self.fit_options)
            self.update_image()

    def update_image(self):
        indmin, indmax = np.searchsorted(self.energy, self.edge.e_int)

        if self.si_bsub is None:
            self.im_inel = np.mean(self.si[:,:,indmin:indmax],axis=(-1))
            self.h['inel'].set_array(self.im_inel)
            self.h['inel'].autoscale()
        else:           
            self.im_inel = np.mean(self.si_bsub[:,:,indmin:indmax],axis=(-1))
            self.h['inel'].set_array(self.im_inel)
            self.h['inel'].autoscale()


    def dummy(self, *args):
        pass

    # def add_save_dict(event):
    #     results_dict['bsub_spectrum'] = bsub
    #     results_dict['image'] = inel_im
    #     results_dict['bsub_SI'] = bsub_array
    #     results_dict['edge'] = [fit_window[0],fit_window[1],int_window[0],int_window[1]]
 