import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, RectangleSelector, CheckButtons
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, tqdm_notebook
import spectrum_image.SI_lineshapes as ls

# from lmfit import Parameters, model, Minimizer, minimize
# from lmfit.models import PowerLawModel




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
    
        
