## Intergrid: interpolate data given on an N-d rectangular grid

Purpose: interpolate data given on an N-d rectangular grid,
uniform or non-uniform,
using the fast
[scipy.ndimage.map_coordinates](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.interpolation.map_coordinates.html).
Non-uniform grids are first uniformized with
[numpy.interp](https://docs.scipy.org/doc/numpy/reference/generated/numpy.interp.html).

Keywords, tags: interpolation, rectangular grid, box grid, python, numpy, scipy

Background:
the reader should know some Python and NumPy
([IPython](https://ipython.org) is invaluable for learning both).
For basics of interpolation, see
[Bilinear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation)
on Wikipedia. For `map_coordinates`, see the example under
[multivariate-spline-interpolation-in-python-scipy](https://stackoverflow.com/questions/6238250/multivariate-spline-interpolation-in-python-scipy)
on stackoverflow.

### Example

Say we have rainfall on a 4 x 5 grid of rectangles, lat 52 .. 55 x lon -10 .. -6,
and want to interpolate (estimate) rainfall at 1000 query points
in between the grid points.

	from intergrid.intergrid import Intergrid  # .../intergrid/intergrid.py

        # define the grid --
    griddata = np.loadtxt(...)  # griddata.shape == (4, 5)
    lo = np.array([ 52, -10 ])  # lowest lat, lowest lon
    hi = np.array([ 55, -6 ])   # highest lat, highest lon

        # set up an interpolator function "interfunc()" with class Intergrid --
    interfunc = Intergrid( griddata, lo=lo, hi=hi )

        # generate 1000 random query points, lo <= [lat, lon] <= hi --
    query_points = lo + np.random.uniform( size=(1000, 2) ) * (hi - lo)

        # get rainfall at the 1000 query points --
    query_values = interfunc.at( query_points )  # -> 1000 values

What this does: for each [lat, lon] in query_points,

1) find the square of `griddata` it's in,
e.g. [52.5, -8.1] -> [0, 3] [0, 4] [1, 4] [1, 3]\
2) do bilinear (multilinear) interpolation in that square,
using `scipy.ndimage.map_coordinates` .

Check:\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `interfunc( lo ) == griddata[0, 0]`\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `interfunc( hi ) == griddata[-1, -1]` i.e. `griddata[3, 4]`


### Parameters

`griddata`: numpy array_like, 2d 3d 4d ...\
`lo, hi`: user coordinates of the corners of griddata, 1d array-like, lo < hi\
`maps`: an optional list of `dim` descriptors of piecewise-linear or nonlinear maps,\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; e.g. [[50, 52, 62, 63], None] \ \ # uniformize lat, linear lon; see below\
`copy`: make a copy of query_points, default `True`;\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `copy=False` overwrites query_points, runs in less memory\
`verbose`: the default 1 prints a summary of each call, with run time\
`order`: interpolation order:\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; default 1: bilinear, trilinear ... interpolation using all 2^dim corners\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 0: each query point -> the nearest grid point -> the value there\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2 to 5: spline, see below\
`prefilter`: the kind of spline:\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; default `False`: smoothing B-spline\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `True`: exact-fit C-R spline\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1/3: Mitchell-Netravali spline, 1/3 B + 2/3 fit


### Methods

After setting up `interfunc = Intergrid(...)`, either

    query_values = interfunc.at( query_points )  # or
    query_values = interfunc( query_points )

do the interpolation. (The latter is `__call__` in python.)


### Non-uniform rectangular grids

What if our griddata above is at non-uniformly-spaced latitudes,
say [50, 52, 62, 63] ?  `Intergrid` can "uniformize" these
before interpolation, like this:

    lo = np.array([ 50, -10 ])
    hi = np.array([ 63, -6 ])
    maps = [[50, 52, 62, 63], None]  # uniformize lat, linear lon
    interfunc = Intergrid( griddata, lo=lo, hi=hi, maps=maps )

This will map (transform, stretch, warp) the lats in query_points column 0
to array coordinates in the range 0 .. 3, using `np.interp` to do
piecewise-linear (PWL) mapping:

    50  51  52  53  54  55  56  57  58  59  60  61  62  63   # lo[0] .. hi[0]
     0  .5   1  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9  2   3

`maps[1] None` says to map the lons in query_points column 1 linearly:

    -10  -9  -8  -7  -6   # lo[1] .. hi[1]
      0   1   2   3   4

Mapping details

The query_points are first clipped, then columns mapped linearly or non-linearly,
then fed to `map_coordinates` .\
In `dim` dimensions (i.e. axes or columns), `lo` and `hi` are each `dim` numbers,
the low and high corners of the data grid.\
`maps` is an optional list of `dim` map descriptors, which can be

* `None`: linear-transform that column, `query_points[:,j]`, to `griddata`:\
    `lo[j] -> 0`\
    `hi[j] -> griddata.shape[j] - 1`
* a callable function: e.g. `np.log` does\
    `query_points[:,j] = np.log( query_points[:,j] )`
* a *sorted* array describing a non-uniform grid:\
    `query_points[:,j] =`\
    `np.interp( query_points[:,j], [50, 52, 62, 63], [0, 1, 2, 3] )`


### Download

    git clone https://github.com/denis-bz/intergrid.git
        # ? pip install --user git+https://github.com/denis-bz/intergrid.git
        # ? pip install --user intergrid

    # tell python where the intergrid directory is, e.g. in your ~/.bashrc:
    #   export PYTHONPATH=$PYTHONPATH:.../intergrid/

    # test in python or IPython:
    from intergrid.intergrid import Intergrid  # i.e. .../intergrid/intergrid.py


### Splines

`Intergrid( ... order = 0 to 5 )` gives the spline order:\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `order=1`, the default, does bilinear, trilinear ...
interpolation, which looks at the grid data at all 4 8 16 .. 2^dim corners of
the box around each query point.\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `order=0` looks at only the one gridpoint
nearest each query point &mdash; crude but fast.\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `order = 2 to 5` does spline interpolation on a uniform
or uniformized grid, looking at (order+1)^dim neighbors of each query point.

`Intergrid( ... prefilter = False | True | 1/3 )`
specifies the kind of spline, for `order >= 2`:\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `prefilter=0` or `False`, the default: B-spline\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `prefilter=1` or `True`: exact-fit spline\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; `prefilter=1/3`: M-N spline.\
A B-spline goes through smoothed data points,
with [1 4 1] smoothing, [0 0 1 0 0] -> [0 1 4 1 0] / 6.\
An exact-fit a.k.a interpolating spline
goes through the data points exactly.
This is not what you want for noisy data,
and may also wiggle or overshoot more than B-splines do.\
An M-N spline blends 1/3 B-spline and 2/3 exact-fit; see
Mitchell and Netravali,
[Reconstruction filters in computer-graphics](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.125.201&rep=rep1&type=pdf) ,
1988, and the plots from `intergrid/test/MNspline.py`.

<small>
Fine print: Exact-fit or interpolating splines can be local or global.
Catmull-Rom splines and the original M-N splines are local:
they look at 4 neighbors of each query point in 1d, 16 in 2d, 64 in 3d.
Prefiltering is global, with IIR falloff ~ 1 / 4^distance.
(I don't know of test images that show a visible difference to local C-R).
Confusingly, the term "Cardinal spline" is sometimes used
for local (C-R, FIR),
and sometimes for global (IIR prefilter, then B-spline).

Prefiltering is a clever transformation
such that `Bspline( transform( data )) = exactfitspline( data )`.
It is described in a paper by M. Unser,
[Splines: A perfect fit for signal and image processing](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.6706&rep=rep1&type=pdf) ,
1999.
</small>

Uniformizing a grid with PWL, then uniform-splining, is fast and simple,
but not as smooth as true splining on the original non-uniform grid.
The differences will of course depend on the grid spacings
and on how rough the function is.


### Notes

Run any interpolator on *your* data with orders 0, 1 ...
to get an idea of how the results get smoother, and take longer.
Check a few query points by hand; plot some cross-sections.

`griddata` values can be of any numpy integer or floating type: int8 uint8
.. int32 int64 float32 float64.
Beware of overflow: interpolating uint8 s can give values outside the range 0 .. 255.
(Interpolation in `d` dimensions can overshoot by (9/8)^d .)
`np.float32` will use less memory than `np.float64`,
but beware of functions in the flow that silently convert everything
to float64.  The values must be numbers, not vectors.

Coordinate scaling doesn't matter to `Intergrid`;
corner weights are calculated in unit cubes of `griddata`,
after scaling and mapping. If for example griddata column 3
is multiplied by 1000, and lo[3] hi[3] too, the weights are unchanged.

Box grids get big and slow above 5d.
A cube with steps 0 .1 .2 .. 1.0 in all dimensions
has 11^6 ~ 1.8M points in 6d, 11^8 ~ 200M in 8d.
One can reduce that only with a coarser grid like 0 .5 1 in some dimensions
(those that vary the least).
But time ~ 2^d per query point grows pretty fast.

`map_coordinates` in 5d with `order=1` looks at 32 corner values, with average weight 3 %.
If the weights are roughly equal
(which they will tend to be, by the central limit theorem ?),
sharp edges or gradients will be blurred, and colors mixed to a grey fog.

To see how different interpolators affect images, run matplotlib
`plt.imshow( interpolation = "nearest" / "bilinear" / ... )` .


### Kinds of grids

Terminology varies, so the basic kinds of box grids
a.k.a. rectangular grids are defined here.

An integer or Cartesian grid has integer coordinates,
e.g.  2 x 3 x 5 points in a numpy array:
`A = np.array((2,3,5)); A[0,0,0], A[0,0,1] .. A[1,2,4]`.

A uniform box grid has nx x ny x nz ... points uniformly spaced,
linspace x linspace x linspace ...
so all boxes have the same size and are axis-aligned.
Examples: 1024 x 768 pixels on a screen,
or 4 x 5 points at latitudes [10 20 30 40] x longitudes [-10 -9 -8 -7 -6].

A non-uniform box grid also has nx x ny x nz ... points,
but allows non-uniform spacings,
e.g. latitudes [-10 0 60 70] x longitudes [-10 -9 0 20 40];
the boxes have different sizes but are still axis-aligned.

(Scattered data, as the name says, has points anywhere,
not only on grid lines.
To interpolate scattered data in `scipy`, see
[scipy.interpolate.griddata](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html)
and
[scipy.spatial.cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)
.)

There are countless varieties of grids:
grids with holes, grids warped to various map projections,
multiscale / multiresolution grids ...


### Run times

See intergrid/test/test-4d.py: a 4d grid with 1M scattered query points,
uniform / non-uniform box grid, on a 2.5Gz i5 iMac:

    shape (361, 720, 47, 8)  98M * 8
    Intergrid: 617 msec  1000000 points in a (361, 720, 47, 8) grid  0 maps  order 1
    Intergrid: 788 msec  1000000 points in a (361, 720, 47, 8) grid  4 maps  order 1


### See also

[scipy.ndimage.interpolation.map_coordinates](https://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.ndimage.interpolation.map_coordinates.html)\
[scipy reference ndimage](https://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html)\
[stackoverflow.com/questions/tagged/scipy+interpolation](https://stackoverflow.com/questions/tagged/scipy+interpolation)\
[interpol 2014](https://github.com/denis-bz/interpol) -- intergrid + barypol\
Google "regrid | resample"\
`pip search interpol` (also gets string interpolation)


### Comments welcome

and testcases most welcome\
    &mdash; denis-bz-py at t-online dot de

