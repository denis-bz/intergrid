#!/usr/bin/env python
# test-interpol.py dim= side= ...
# for testfunc in [...]:
# for method in [Intergrid ...]:
#   |interpolated - exact| uniform-random query pointsin side^dim

from __future__ import division, print_function
import sys
from time import time
import numpy as np
import scipy

from intergrid.intergrid import Intergrid  # .../intergrid/intergrid.py
from testfuncs import testfuncs  # [ cos_xav ... on unit cube ]
import util as ut

np.set_printoptions( threshold=20, edgeitems=10, linewidth=140,
        formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g

print(80 * "=")
print("versions: numpy %s  scipy %s  python %s" % (
    np.__version__, scipy.__version__, sys.version.split()[0] ))

#...............................................................................
methods = [Intergrid]
dim = 3
side = 20  # uniform grid side^dim
nquery = 1e6
jmethod = -1
jfunc = -1
dtype = np.float64  # map_coord only
orders = [1, 2]
save = 0
seed = 0

# to change these params, run this.py  a=1  b=None  'c = expr' ... in sh or ipython --
for arg in sys.argv[1:]:
    exec( arg )

np.random.seed(seed)

nquery = int(nquery)
if jmethod >= 0:
    methods = [methods[jmethod]]
if jfunc >= 0:
    testfuncs = [testfuncs[jfunc]]
shape = dim * [side]

    # side^d x d  000 .. 111 --
grid = np.array( list( np.ndindex(*shape) ), dtype ) / (side - 1)
lo = np.array( dim * [0.] )
hi = np.array( dim * [1.] )
query = np.random.uniform( size=(nquery, dim) ).astype(dtype)

#...............................................................................
params = "%s  shape %s  nquery %d" % (__file__, shape, nquery)
print( params )

for testfunc in testfuncs:
    funcname = testfunc.__name__
    funcgrid = testfunc( grid ).reshape(shape)
    print("")
    exact = testfunc(query)

    for method in methods:
        methodname = method.__name__
        query_out = np.empty( nquery )

        for order in orders:
            interpolator = method( funcgrid, lo=lo, hi=hi, order=order, verbose=0 )
            t0 = time()  # wallclock

            interpolator.at( query, query_out )
            ms = (time() - t0) * 1000

            diffs = ut.avmaxdiff( query_out, exact, query )
            print( "%4d ms  |%s - exact|  %s  order %d: %s " % (
                    ms, methodname, funcname, order, diffs ))

            if save:  # to plot
                sav = "%dd-order%d-%s-%s.npz" % (dim, order, methodname, funcname)
                print( "np.savez >", sav )
                np.savez( sav, query_out=query_out.astype(np.float32),
                        diffs=diffs, params=params )

