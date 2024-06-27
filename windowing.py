#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np
import numba as _nb

from special_functions import _Tukey

@_nb.guvectorize([(_nb.float64[:],_nb.float64[:],_nb.float64,_nb.float64,_nb.float64,_nb.float64,_nb.float64[:])], '(n),(n),(),(),(),()->(n)')
def Tukey(Y,x,x_1,x_2,x_3,x_4,res):
	res[:] = Y[:]*_Tukey(x[:],x_1,x_2,x_3,x_4)
    