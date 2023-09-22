#!/bin/env/python
#! -*- coding: utf-8 -*-

import numpy as _np
from numba import float64,complex128,vectorize,guvectorize

@vectorize([float64(float64,float64,float64,float64,float64)])
def _Tukey(x,x_1,x_2,x_3,x_4):
    if (x_1<=x) and (x < x_2) :
        return 0.5 - 0.5*_np.cos( _np.pi*(x-x_1)/(x_2-x_1) )
    elif ( (x_2<=x) and (x < x_3) ) :
        return 1.0 
    elif ( (x_3<=x) and (x < x_4) ) :
        return 0.5 - 0.5*_np.cos( _np.pi*(x_4-x)/(x_4-x_3) );
    else :
        return 0.0 
@guvectorize([(float64[:],float64,float64,float64,float64,float64[:])], '(n),(),(),(),()->(n)')
def Tukey(x,x_1,x_2,x_3,x_4,res):
    """
    Tukey(x,x_1,x_2,x_3,x_4)
        0 ... cos(-pi/2)->cos(0) 1 ... 1 cos(0)->cos(-pi/2)  0 ...
          ....x_1       ->   x_2         x_3       ->   x_4  ...
    """
    res[:] = _Tukey(x[:],x_1,x_2,x_3,x_4)