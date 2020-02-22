#!/usr/bin/env python
# testfuncs.py: [ f( X npt x dim in unit cube ) ... ]
# scaled so all max grad ~ 1  for dim 4 side 20

from __future__ import division, print_function
import numpy as np
from numpy import pi, cos, sin

np.set_printoptions( threshold=20, edgeitems=10, linewidth=140,
        formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g

#...............................................................................
def cos_xav( X, freq=1, **kw ):
    """ cos 2 pi f Xav, X in unit cube
        Xav ~ sqrt(dim) / 2
        freq 1 dim 4:
            axis 1 .. 0   cos 2 pi xav = x / dim
            diag 1 .. -1 .. 1   cos 2 pi xav = x
    """
    return cos( 2 * pi * freq * X.mean(axis=1) ) / .004

def volripple( X, fm=6, alpha=.25, **kw ):
    """ npt x dim -> npt
    from Marschner & Lobb, reconstruction filters for volume rendering
    side 20 / h .05 in 3d:
    """
    X = np.asanyarray(X)
    Xdim = X.ndim
    if Xdim == 1:
        X = np.array([X])
    r = np.sqrt( (X[:,:-1] ** 2) .sum(axis=1) )
    z = X[:,-1]
    coscos = cos( 2*pi*fm * cos( pi * r / 2 ))
    rho = (1 - sin( pi * z / 2 ) + alpha * (1 + coscos)) \
        / (2 * (1 + alpha))
    return (rho if Xdim == 2  else rho[0]) / .01

def linear_should_be_exact( X ):
    return X.mean(axis=1)

def parabola( X ):
    xav = X.mean(axis=1)
    return (xav**2 - xav) / .0006

def sawtooth( X, c=.1 ):
    return (c * X.sum(axis=1)) % 1


testfuncs = [cos_xav, volripple, parabola]

#...............................................................................
if __name__ == "__main__":
    import sys

    dim = 3
    side = 20  # grid side^dim

    # to change these params, run this.py  a=1  b=None  'c = expr' ... in sh or ipython --
    for arg in sys.argv[1:]:
        exec( arg )

    side |= 1
    shape = dim * [side]
    grid = np.array( list( np.ndindex(*shape) ), float ) / (side - 1)

    for func in testfuncs:
        print("\n%s --" % func.__name__)
        funcgrid = func( grid ).reshape(shape)
        # print funcgrid
        absgrad = np.fabs( np.gradient( funcgrid, *shape ))  # dim, shape
        jmax = absgrad.argmax()
        # at = np.unravel_index( jmax, absgrad.shape )
        print("max grad: %.2g at %s" % (
            absgrad.ravel()[jmax], grid[ jmax // dim ]))

