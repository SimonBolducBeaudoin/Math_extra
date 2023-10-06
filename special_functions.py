#!/bin/env/python
#! -*- coding: utf-8 -*-

import numpy as _np
import numba as _nb

from integrate import NewtonCotes11

@_nb.vectorize([_nb.float64(_nb.float64,_nb.float64,_nb.float64,_nb.float64,_nb.float64)])
def _Tukey(x,x_1,x_2,x_3,x_4):
    if (x_1<=x) and (x < x_2) :
        return 0.5 - 0.5*_np.cos( _np.pi*(x-x_1)/(x_2-x_1) )
    elif ( (x_2<=x) and (x < x_3) ) :
        return 1.0 
    elif ( (x_3<=x) and (x < x_4) ) :
        return 0.5 - 0.5*_np.cos( _np.pi*(x_4-x)/(x_4-x_3) );
    else :
        return 0.0 
@_nb.guvectorize([(_nb.float64[:],_nb.float64,_nb.float64,_nb.float64,_nb.float64,_nb.float64[:])], '(n),(),(),(),()->(n)')
def Tukey(x,x_1,x_2,x_3,x_4,res):
    """
    Tukey(x,x_1,x_2,x_3,x_4)
        0 ... cos(-pi/2)->cos(0) 1 ... 1 cos(0)->cos(-pi/2)  0 ...
          ....x_1       ->   x_2         x_3       ->   x_4  ...
    """
    res[:] = _Tukey(x[:],x_1,x_2,x_3,x_4)

@_nb.vectorize([_nb.float64(_nb.float64)])
def C(x):
	return _np.cos((_np.pi/2.0)*x*x)
@_nb.vectorize([_nb.float64(_nb.float64)])
def S(x):
	return _np.sin((_np.pi/2.0)*x*x)

@_nb.vectorize([_nb.float64(_nb.float64)])
def FresnelCos(x):
	return NewtonCotes11(C,0,x,1000)
    
@_nb.vectorize([_nb.float64(_nb.float64)])
def FresnelSin(x):
	return NewtonCotes11(S,0,x,1000)