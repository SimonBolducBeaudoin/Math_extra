#!/bin/env/python
#! -*- coding: utf-8 -*-
from numba import njit

@njit
def NewtonCotes11( f ,lower_bound=0,upper_bound=1,N=1000):
	h = (upper_bound-lower_bound)/(10*N);
	I_1 = 0
	I_2 = 0
	I_3 = 0
	I_4	= 0
	I_5	= 0
	I_6	= 0
	prefact = 5.0/299376.0;
	k1 = 16067;
	k2 = 106300;
	k3 = -48525;
	k4 = 272400
	k5 = -260550;
	k6 = 427368;
	x = lower_bound 
	for i in range(N):
		I_1 += f( x       ) + f( x + 10*h )
		I_2 += f( x + h   ) + f( x + 9 *h )
		I_3 += f( x + 2*h ) + f( x + 8 *h )
		I_4 += f( x + 3*h ) + f( x + 7 *h )
		I_5 += f( x + 4*h ) + f( x + 6 *h )
		I_6 += f( x + 5*h )
		x += 10*h
	return prefact*h*( k1*I_1 + k2*I_2 + k3*I_3 + k4*I_4 + k5*I_5 + k6*I_6 )

