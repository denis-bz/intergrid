# intergrid/test/util.py

from __future__ import division
import numpy as np

def avmaxdiff( interpol, exact, query_points=None ):
    absdiff = np.fabs( interpol - exact )
    av = absdiff.mean()
    jmax = absdiff.argmax()  # flat
    at = "at %s" % query_points[jmax]  if query_points is not None \
        else ""
    return "av %.2g  max %.2g = %.3g - %.3g  %s" % (
            av, absdiff[jmax], interpol[jmax], exact[jmax], at )

