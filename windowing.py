#!/bin/env/python
#! -*- coding: utf-8 -*-
import numpy as _np
from numba import float64, guvectorize

from special_function import Tukey as Tukey_func

@guvectorize([(float64[:],float64[:],float64,float64,float64,float64,float64[:])], '(n),(n),(),(),(),()->(n)')
def Tukey(Y,x,x_1,x_2,x_3,x_4,res):
	res[:] = Y[:]*Tukey_func(x[:],x_1,x_2,x_3,x_4)
    