import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, RectangleSelector, SpanSelector, CheckButtons, RadioButtons, Button, TextBox
from matplotlib.backend_bases import MouseButton


class EnergyMap :
    def __init__( self, si, einc=None, eloss=None):

        si[np.isnan(si)] = 0
        self.si = si
        (self.neloss,  self.neinc) = self.si.shape


        if eloss is None:
            eloss = np.arange( self.neloss )
        if einc is None:
            einc = np.arange( self.neinc )

        self.eloss = eloss
        self.einc = einc

        ind_eloss = np.argsort( self.eloss )
        ind_einc  = np.argsort( self.einc )

        self.eloss = self.eloss[ind_eloss]
        self.einc = self.einc[ind_einc]

        self.si = self.si[ind_eloss,:]
        self.si = self.si[:,ind_einc]

    def browser( self, cmap='gray', figsize=(6,8), vmin=None, vmax=None):
        
        self.int_dir = 0
        self.eaxis  = [self.einc, self.eloss]
        self.elabel = ["Incident Energy (eV)", "Energy Loss (eV)"]
        ## Initialize browser object
        self.spec1 = np.mean(self.si,axis=self.int_dir)
        self.spec2 = np.mean(self.si,axis=self.int_dir)

        self.im_inel = self.si


        ##############Set Initial plot#################
        self.fig=plt.figure(figsize=figsize,layout='constrained')

        self.ax = {}
        self.ax['inel']=self.fig.add_axes([0.1,0.55,0.8,0.4]) # Image
        self.ax['spec']=self.fig.add_axes([0.1,0.05,0.8,0.4]) # Spectrum
        self.ax['ck_ysetting'] = self.fig.add_axes([0.7,0.45,0.3,0.07]) # Y-lock chkbox
        self.ax['ck_roisetting'] = self.fig.add_axes([0.7,0.94,0.3,0.07]) # ROI2 chkbox

        ## Initialize plot handles
        self.h = {}
        ################## ax['inel'] ######################
        self.h['inel'] = self.ax['inel'].pcolormesh( self.einc, self.eloss, self.im_inel,cmap = cmap, vmin=vmin, vmax=vmax)
        self.ax['inel'].set_axis_on()
        self.ax['inel'].set_title('RIXS Energy Map')
        self.ax['inel'].set_ylabel('Energy Loss (eV)')
        self.ax['inel'].set_xlabel('Incident Energy (eV)')
        self.ax['inel'].set_xlim( (np.min(self.einc),np.max(self.einc)))
        self.ax['inel'].set_ylim( (np.min(self.eloss),np.max(self.eloss)))
        # self.ax['inel'].autoscale(enable=True, axis='xy', tight=True)

        ################## ax['spec'] ######################
        self.h['spec1'], = self.ax['spec'].plot(self.eaxis[self.int_dir], self.spec1,color='crimson')
        self.h['spec2'], = self.ax['spec'].plot(self.eaxis[self.int_dir], self.spec2,color='royalblue')
        self.h['spec2'].set_alpha(0)
        # self.ax['spec'].set_yticks([])
        self.ax['spec'].set_xlabel(self.elabel[self.int_dir])
        self.ax['spec'].set_ylabel('Intensity')


        ## Initialize ui handles
        self.ui={}
        ### Check box 
        # ylim locker
        self.ui['ck_ysetting'] = CheckButtons(ax=self.ax['ck_ysetting'], labels= ["Lock Y-axis","Log(y)"],
                                            actives=[False, False], check_props={'facecolor': 'k'} )
        self.ui['ck_ysetting'].on_clicked( lambda v: self.onclick_ck_ysetting() )
        self.y_locked = False
        self.y_log = False


        self.ui['ck_roisetting'] = CheckButtons(ax=self.ax['ck_roisetting'], labels= ["Enable ROI 2", "Integrate E-loss"],
                                        actives=[False, False], check_props={'facecolor': 'k'} )
        self.ui['ck_roisetting'].on_clicked( lambda v: self.onclick_ck_roisetting() )
        self.roi2_enabled = False


        ################### Selectors ###################
        self.ui['roi1'] = RectangleSelector(self.ax['inel'], self.dummy, button=[1],
                                        useblit=True ,minspanx=1, minspany=1,spancoords='pixels',
                                        interactive=True,props=dict(facecolor='crimson',edgecolor='crimson',alpha=0.2,fill=True),
                                        handle_props=dict(markersize=2,markerfacecolor='white'))#,ignore_event_outlpde=True
        
        self.ui['roi2'] = RectangleSelector(self.ax['inel'], self.dummy, button=[3],
                                        useblit=True ,minspanx=1, minspany=1,spancoords='pixels',
                                        interactive=True,props=dict(facecolor='royalblue',edgecolor='royalblue',alpha=0.2,fill=True),
                                        handle_props=dict(markersize=2,markerfacecolor='white'))#,ignore_event_outlpde=True)   
        self.ui['roi2'].set_visible( False )
        self.ui['roi2'].set_active( False )
            

        self.fig.canvas.mpl_connect( 'motion_notify_event', 
                                    lambda event: self.onclick_figure(event))
        

        # self.rescale_yrange()
        # return results_dict,selector_collection
        
    ################### Update Functions ###################

    
    def onclick_ck_ysetting(self):
        self.y_locked = self.ui['ck_ysetting'].get_status()[0]
        self.y_log = self.ui['ck_ysetting'].get_status()[1]
        self.rescale_yrange()
    
    def onclick_ck_roisetting(self):
        # Check for ROI2 
        self.roi2_enabled = self.ui['ck_roisetting'].get_status()[0]
        if self.roi2_enabled:
            self.h['spec2'].set_alpha(1)
            self.ui['roi2'].set_visible( True )
            self.ui['roi2'].set_active( True )
        else:
            self.h['spec2'].set_alpha(0)
            self.ui['roi2'].set_visible( False )
            self.ui['roi2'].set_active( False )


        # Check for integration flip
        if self.ui['ck_roisetting'].get_status()[1]:
            self.int_dir = 1
            self.ui['ck_roisetting'].labels[1].set_text('Integrate E-inc')
            self.ax['spec'].set_xlabel('Energy Loss (eV)')
        else:
            self.int_dir = 0
            self.ui['ck_roisetting'].labels[1].set_text('Integrate E-loss')
            self.ax['spec'].set_xlabel('Incident Energy (eV)')

        self.int_dir = int(self.ui['ck_roisetting'].get_status()[1])
        self.on_change_roi1()
        # self.on_change_roi2()


    def on_change_roi1(self):
        roi1 = self.ui['roi1'].extents
        roi2 = self.ui['roi2'].extents
        if self.int_dir == 0:
            self.ui['roi2'].extents = (roi1[0],roi1[1],roi2[2],roi2[3])
        else:
            self.ui['roi2'].extents = (roi2[0],roi2[1],roi1[2],roi1[3])

        eimin = np.searchsorted( self.einc,  float(roi1[0]))
        eimax = np.searchsorted( self.einc,  float(roi1[1]))
        elmin = np.searchsorted( self.eloss, float(roi1[2]))
        elmax = np.searchsorted( self.eloss, float(roi1[3]))

        if eimin == eimax:
            eimax += 1
        if elmin == elmax:
            elmax += 1


        if self.int_dir == 0:
            self.spec1  = np.mean(self.si[elmin:elmax,:],axis=(0))
            eminmax = (self.einc[eimin], self.einc[eimax])
        else:
            self.spec1 = np.mean(self.si[:,eimin:eimax],axis=(1))
            eminmax = (self.eloss[elmin], self.eloss[elmax])
        
        self.ax['spec'].set_xlim( eminmax )
        self.h['spec1'].set_data( self.eaxis[self.int_dir], self.spec1)



        self.rescale_yrange()

    def on_change_roi2(self):
        roi1 = self.ui['roi1'].extents
        roi2 = self.ui['roi2'].extents

        if self.int_dir == 0:
            self.ui['roi2'].extents = (roi1[0],roi1[1],roi2[2],roi2[3])
        else:
            self.ui['roi2'].extents = (roi2[0],roi2[1],roi1[2],roi1[3])

        eimin = np.searchsorted( self.einc,  float(roi2[0]))
        eimax = np.searchsorted( self.einc,  float(roi2[1]))
        elmin = np.searchsorted( self.eloss, float(roi2[2]))
        elmax = np.searchsorted( self.eloss, float(roi2[3]))
        if eimin == eimax:
            eimax += 1
        if elmin == elmax:
            elmax += 1

        if self.int_dir == 0:
            self.spec2  = np.mean(self.si[elmin:elmax,:],axis=(0))
        else:
            self.spec2 = np.mean(self.si[:,eimin:eimax],axis=(1))
        

        self.h['spec2'].set_data( self.eaxis[self.int_dir], self.spec2)

        self.rescale_yrange()
    


    def rescale_yrange(self):

        if self.y_log:
            self.ax['spec'].set_yscale('log')
            self.ax['spec'].set_ylabel('Log Intensity')
        else:
            self.ax['spec'].set_yscale('linear')
            self.ax['spec'].set_ylabel('Intensity')

        if self.y_locked == False:

            roi1 = self.ui['roi1'].extents
            eimin = np.searchsorted( self.einc,  int(roi1[0]))
            eimax = np.searchsorted( self.einc,  int(roi1[1]))
            elmin = np.searchsorted( self.eloss, int(roi1[2]))
            elmax = np.searchsorted( self.eloss, int(roi1[3]))
            if eimin == eimax:
                eimax += 1
            if elmin == elmax:
                elmax += 1


            if not self.y_log:
                if self.int_dir == 1:
                    maxval =  1.02*self.spec1.max()
                    minval =  0.98*self.spec1.min()
                else:
                    maxval =  1.02*self.spec1.max()
                    minval =  0.98*self.spec1.min()

            else:
                if self.int_dir == 1:
                    maxval =  1.1*self.spec1.max()
                    minval =  0.9*self.spec1.min()
                else:
                    maxval =  1.1*self.spec1.max()
                    minval =  0.9*self.spec1.min()



            if self.roi2_enabled:
                if not self.y_log:
                    if self.int_dir == 1:
                        maxval2 =  1.02*self.spec2.max()
                        minval2 =  0.98*self.spec2.min()
                    else:
                        maxval2 =  1.02*self.spec2.max()
                        minval2 =  0.98*self.spec2.min()

                else:
                    if self.int_dir == 1:
                        maxval2 =  1.1*self.spec2.max()
                        minval2 =  0.9*self.spec2.min()
                    else:
                        maxval2 =  1.1*self.spec2.max()
                        minval2 =  0.9*self.spec2.min()

                minval = min( minval, minval2)
                maxval = max( maxval, maxval2)

            # print( minval, maxval )

            self.ax['spec'].set_ylim([minval,maxval])


    ############### Event Handlers ###################
    def onclick_figure( self, event ):
        if event.inaxes in [self.ax['inel']]:
            if event.button == MouseButton.LEFT:
                # Left Click on Inelastic Image
                self.on_change_roi1()
            elif event.button == MouseButton.RIGHT:
                if self.roi2_enabled:
                    # Right Click on Inelastic Image
                    self.on_change_roi2()

        # elif event.inaxes in [self.ax['spec']]:
        #     if event.button == MouseButton.LEFT:
        #         self.fit_check = True
        #         self.ax['e_bsub'].set_visible(True)
        #         self.ui['slid_e_bsub'].set_val( self.ui['bsub'].extents )

        #     elif event.button == MouseButton.RIGHT:
        #         self.int_check = True
        #         self.ax['e_int'].set_visible(True)
        #         self.ui['slid_e_int'].set_val( self.ui['int'].extents )
        
        # elif event.inaxes in [self.ax['spec2']]:    
        #     if event.button == MouseButton.LEFT:
        #         self.fit_check = True
        #         self.ax['e_bsub'].set_visible(True)
        #         self.ui['slid_e_bsub'].set_val( self.ui['bsub2'].extents )

        #     elif event.button == MouseButton.RIGHT:
        #         self.int_check = True
        #         self.ax['e_int'].set_visible(True)
        #         self.ui['slid_e_int'].set_val( self.ui['int2'].extents )


    # def update_image(self):
    #     indmin, indmax = np.searchsorted(self.eaxis, self.edge.e_int)
    #     if indmin == indmax:
    #         indmax +1

    #     if self.si_bsub is None:
    #         self.si_inel = np.mean(self.si[:,indmin:indmax],axis=(-1))
    #     else:           
    #         self.si_inel = np.mean(self.si_bsub[:,indmin:indmax],axis=(-1))
        

    #     if not self.adf_enabled:
    #         self.h['plot'].set_ydata(self.si_inel)
    #         self.ax['plot'].relim() 
    #         self.ax['plot'].autoscale_view()


    def dummy(self, *args):
        pass
