import numpy as np

## Peak Like functions
def gaussian( x, A, e0, sg ):
    return A*np.exp( -0.5*( ((x-e0)/sg)**2 ) )
def d_gaussian( x, A, e0, sg ):
    dfdA = np.exp( -0.5*( ((x-e0)/sg)**2 ) )
    dfde = A*(x-e0)/(sg**2)*dfdA
    dfds = (x-e0)/sg*dfdA
    return np.asarray([dfdA,dfde,dfds]).T

def lorentzian( x, A, e0, sg ):
    gm = sg*np.sqrt(2*np.log(2))
    return A/( ((x-e0)/gm)**2 + 1 )
def d_lorentzian( x, A, e0, sg ):
    gm = sg*np.sqrt(2*np.log(2))
    dfdA = 1/( ((x-e0)/gm)**2 + 1 )
    dfde = 2*e0*A*( dfdA**2 )* ((x-e0)/gm)
    dfds = 2*A*( dfdA**2 )*((x-e0)**2)/(gm**3) * np.sqrt(2*np.log(2))
    return np.asarray([dfdA,dfde,dfds]).T

## Decay Functions
def powerlaw( x, A1, r1 ):
    return A1 * ( x**(-r1) )
def d_powerlaw( x, A1, r1):
    dfdA = x**(-r1)
    dfdr = -A1*np.log(x)*dfdA
    return np.array( [dfdA, dfdr]).T

def LCPL( x, A1, r1, A2, r2 ):
    return A1 * ( x**(-r1) ) + A2 * ( x**(-r2) )
def d_LCPL( x, A1, r1, A2,r2):
    dfdA1 = x**(-r1)
    dfdr1 = -A1*np.log(x)*dfdA1
    dfdA2 = x**(-r2)
    dfdr2 = -A2*np.log(x)*dfdA2
    return np.array( [dfdA1, dfdr1, dfdA2, dfdr2]).T