#!/bin/env/python
#! -*- coding: utf-8 -*-
import  numpy as _np

def fourier_transform(F,dt):
    """
    This gives and approximation of F(f) or fourier transform of F(t) if you prefer
    and is not directly the DFT of F(t)    
    """
    return dt*_np.fft.rfft(F)
def ifourier_transform(F,dt,n):
    """
    See Also
    -------
        fourier_transform
    """
    irfft = _np.fft.irfft
    shift = _np.fft.fftshift
    return (1.0/dt)*shift(irfft(F,n=n))

#########################
# Central limit theorem #
#########################

def SE(mu2k,muk,n):
    """ 
        Standard error :
        Voir notes Virally Central limit theorem
        Computation of the standard error for the moment of order K
        mu2k : is the moment of order 2 k
        muk  : is the moment of order k
        If these moments are not centered then the definition is good for none centered moment
        Idem for centered moment
    """
    return _np.sqrt(_np.abs(mu2k-muk**2)/float(n))  
