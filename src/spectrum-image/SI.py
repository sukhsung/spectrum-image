import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, RectangleSelector, CheckButtons
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, tqdm_notebook

def testfunc():
    print("hi'")


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

    def PSI_viewer( self, bg_type=None):
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


    
    def PSI_integrator( self ):
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
            self.ax_spec.set_ylim( [-0.1*max_spec, 1.2*max_spec])
        
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
        # print( self.integrate_SI()  )
        im_int,dummy = self.integrate_SI()
        self.h_int.set_data( self.normalize( im_int ) )

    def update_bg( self ):
        fit_i_ch = self.eVtoCh( self.e_bsub[0], self.es)
        fit_f_ch = self.eVtoCh( self.e_bsub[1], self.es)

        e_fit = self.es[ fit_i_ch:fit_f_ch]
        y_fit = self.roi_spectrum[fit_i_ch:fit_f_ch] 
        ftol = np.max(self.roi_spectrum)*1e-5
        
        bg_func, p0, bounds = self.prepare_bgfit( y_fit, e_fit )

        try:
            popt_pl,pcov_pl = curve_fit(bg_func, e_fit, y_fit, 
                                        p0=p0,                                
                                        bounds=bounds,
                                        maxfev=1000, method='trf',
                                        ftol= ftol)
            bg_fit = bg_func(self.es, *popt_pl)
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
            r_g =0
        elif r_g >10:
            r_g = 10
        c_g = y_fit[0]*(e_fit[0]**r_g)

        if self.bg_type == "LCPL":
            p0 = (c_g, r_g, c_g, r_g)
            bounds = ([0,0,0,0],[np.inf, 10, np.inf, 10])
            bg_func = self.LCPL
        elif self.bg_type == "powerlaw":
            p0 = (c_g, r_g)
            bounds = ([0,0],[np.inf,10])
            bg_func = self.powerlaw
        return bg_func, p0, bounds

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


    def bgsub_SI( self, e_bsub=None, bg_type="powerlaw", ftol=0.0005, gtol=0.00005,xtol=None):
        if e_bsub is not None:
            self.e_bsub = e_bsub

        self.bg_type = bg_type
        bg_pl_SI = np.zeros_like(self.si)
        fit_i_ch = self.eVtoCh(self.e_bsub[0], self.es)
        fit_f_ch = self.eVtoCh(self.e_bsub[1], self.es)
        e_fit = self.es[ fit_i_ch:fit_f_ch]
    
        ## Fit Mean Spectrum to get a good guess Parameter
        mean_spec = np.mean( self.si, axis=(0,1))
        y_fit = mean_spec[ fit_i_ch:fit_f_ch]
        # ftol = np.max(mean_spec)*1e-5

        bg_func, p0, bounds = self.prepare_bgfit( y_fit, e_fit )

        p0,pcov_pl = curve_fit(bg_func, e_fit, y_fit, 
                                    p0=p0,                                
                                    bounds=bounds,
                                    maxfev=1000, method='trf',
                                    ftol= ftol)
        
        ## perform PL fit and background subtraction on each pixel.
        pbar = tqdm_notebook(total = (self.nx)*(self.ny),desc = "Background subtracting")
        for i in range(self.nx):
            for j in range(self.ny):
                y_fit = self.si[j,i,fit_i_ch:fit_f_ch]

                try:
                    popt_pl,pcov_pl=curve_fit(bg_func, e_fit, y_fit,
                                            maxfev=50000,p0=p0,method='trf',
                                            verbose = 0, ftol=ftol, gtol=gtol,xtol=xtol)
                    

                    bg_fit = bg_func(self.es, *popt_pl)
                except:
                    bg_fit = 0

                bsub = self.si[j,i]-bg_fit
                bsub[0:fit_i_ch] = 0
                bg_pl_SI[j,i,:] = bsub
                pbar.update(1)
                
        return bg_pl_SI, self.e_bsub
    
    def bgsub_SI_lba( self, e_bsub=None, bg_type="powerlaw", gfwhm=5):
        if e_bsub is not None:
            self.e_bsub = e_bsub

        self.bg_type = bg_type
        bg_pl_SI = np.zeros_like(self.si)
        fit_i_ch = self.eVtoCh(self.e_bsub[0], self.es)
        fit_f_ch = self.eVtoCh(self.e_bsub[1], self.es)
        e_fit = self.es[ fit_i_ch:fit_f_ch]

        lba_bg = self.prepare_lba( gfwhm=gfwhm )

        ## Fit Mean Spectrum to get a good guess Parameter
        mean_spec = np.mean( lba_bg, axis=(0,1))
        y_fit = mean_spec[ fit_i_ch:fit_f_ch]
        ftol = np.max(mean_spec)*1e-5

        bg_func, p0, bounds = self.prepare_bgfit( y_fit, e_fit )

        p0,pcov_pl = curve_fit(bg_func, e_fit, y_fit, 
                                    p0=p0,                                
                                    bounds=bounds,
                                    maxfev=1000, method='trf',
                                    ftol= ftol)
        
        ## perform PL fit and background subtraction on each pixel.
        pbar = tqdm_notebook(total = (self.nx)*(self.ny),desc = "Background subtracting")
        for i in range(self.nx):
            for j in range(self.ny):
                y_fit = lba_bg[j,i,fit_i_ch:fit_f_ch]
                popt_pl,pcov_pl=curve_fit(bg_func, e_fit, y_fit,
                                        maxfev=50000,p0=p0,method='trf',
                                        verbose = 0, ftol=ftol)
                

                bg_fit = bg_func(self.es, *popt_pl)
                bsub = self.si[j,i]-bg_fit
                bsub[0:fit_i_ch] = 0
                bg_pl_SI[j,i,:] = bsub
                pbar.update(1)
        
        return bg_pl_SI, self.e_bsub

    def prepare_lba( self, e_bsub=None, gfwhm=5 ):
        if e_bsub is not None:
            self.e_bsub = e_bsub
        self.gwfm = gfwhm

        fit_i_ch = self.eVtoCh( self.e_bsub[0], self.es)
        fit_f_ch = self.eVtoCh( self.e_bsub[1], self.es)


        lba_raw = np.copy( self.si )
        for e_ch in np.arange( fit_i_ch, fit_f_ch ):
            lba_raw[:,:,e_ch] = gaussian_filter( self.si[:,:,e_ch], sigma=gfwhm/2.35)

        lba_normalized = np.copy(lba_raw)
        pbar2 = tqdm_notebook(total = (self.nx * self.ny ),desc = "Normalizing")
        for i in range(self.nx):
            for j in range(self.ny):
                normalize,pcov_pl=curve_fit( self.lba_normalization,
                                            lba_raw[j,i,fit_i_ch:fit_f_ch],
                                            self.si[j,i,fit_i_ch:fit_f_ch])
                lba_normalized[j,i,fit_i_ch:fit_f_ch] = lba_raw[j,i,fit_i_ch:fit_f_ch]*normalize
                pbar2.update(1)

        return lba_normalized
    
    def lba_normalization( self, lba, m ):
        return m*lba
    
    def dummy( self,a,b):
        ""

    def LCPL( self, energy, c1, r1, c2, r2):
        return c1*energy**(-r1) + c2*energy**(-r2)
    def powerlaw( self, energy, c, r):
        return c*energy**(-r)
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



