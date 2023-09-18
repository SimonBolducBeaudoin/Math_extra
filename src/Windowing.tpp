#include "Windowing.h"
template <class FloatType>
void Tukey_Window( FloatType* argument , double alpha, uint N )
{
	double x_max = (double)(N-1) ;
	/*Could be parallelized*/
	for ( uint i = 0 ; i<N ; i++ )
	{
		argument[i] *= Tukey( (double)i , alpha , x_max );
	}
}

template <class FloatType>
void Tukey_Window( FloatType* argument , double* abscisse, uint N , double alpha , double x_max )
{
	/*Could be parallelized*/
	for ( uint i = 0 ; i<N ; i++ )
	{
		argument[i] *= Tukey( abscisse[i] , alpha , x_max );
	}
}

template <class FloatType>
void Tukey_Window( FloatType* argument , double x_1 , double x_2 , uint N )
{
	double x_max = (double)(N-1) ;
	/*Could be parallelized*/
	for ( uint i = 0 ; i<N ; i++ )
	{
		argument[i] *= Tukey( (double)i , x_1 , x_2 , x_max );
	}
}

template <class FloatType>
void Tukey_Window( FloatType* argument , double* abscisse , uint N , double x_1 , double x_2 , double x_max )
{
	/*Could be parallelized*/
	for ( uint i = 0 ; i<N ; i++ )
	{
		argument[i] *= Tukey( abscisse[i] , x_1 , x_2 , x_max );
	}
}

template <class FloatType>
void Tukey_Window( FloatType* argument , double* abscisse , uint N , double x_1 , double x_2 , double x_3 , double x_4 , double x_max )
{
	/*Could be parallelized*/
	for ( uint i = 0 ; i<N ; i++ )
	{
		argument[i] *= Tukey( abscisse[i] ,x_1,x_2,x_3,x_4,x_max );
	}
}
