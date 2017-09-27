/*******************************************************************************
* PROGRAM: canny_edge_detector
* FILE: x_y_calc.cu 
* NAME: Vuong Pham-Duy
*       Faculty of Computer Science and Technology
*       Ho Chi Minh University of Technology, Viet Nam
*       vuongpd95@gmail.com
* DATE: 11/10/2016*******************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VERBOSE 1

__global__ void derrivative_x_y_kernel(int rows, int cols, int blockSize,
	short int *d_smoothedim, short int *d_delta_x, short int *d_delta_y);

__global__ void magnitude_x_y_kernel(int rows, int cols, int blockSize,
	short int *d_magnitude, short int *d_delta_x, short int *d_delta_y);

/*******************************************************************************
* PROCEDURE: x_y_calc
* PURPOSE: calculate delta_x, delta_y and magnitude of the image
* NAME: Vuong Pham-duy
* DATE: 10/11/2016
*******************************************************************************/
void x_y_calc(int rows, int cols, int blockSize, int gridSize,
	short int **d_delta_x, short int **d_delta_y, short int **d_smoothedim, short int **d_magnitude) 
{
	/****************************************************************************
	* Compute the first derivative in the x and y directions.
	****************************************************************************/
	if (VERBOSE) printf("Computing the X and Y first derivatives.\n");
	derrivative_x_y_kernel<<<gridSize, blockSize>>>(rows, cols, blockSize, (*d_smoothedim), (*d_delta_x), (*d_delta_y));

	/****************************************************************************
	* Compute the magnitude of the gradient.
	****************************************************************************/
	if (VERBOSE) printf("Computing the magnitude of the gradient.\n");
	magnitude_x_y_kernel<<<gridSize, blockSize>>>(rows, cols, blockSize, (*d_magnitude), (*d_delta_x), (*d_delta_y));
}
/*******************************************************************************
* PROCEDURE: derrivative_x_y_kernel
* PURPOSE: Compute the first derivative of the image in both the x any y
* directions. The differential filters that are used are:
*
*                                          -1
*         dx =  -1 0 +1     and       dy =  0
*                                          +1
*
* NAME: Vuong Pham-duy
* DATE: 10/11/2016
*******************************************************************************/
__global__ void derrivative_x_y_kernel(int rows, int cols, int blockSize,
	short int *d_smoothedim, short int *d_delta_x, short int *d_delta_y)
{
	/* This thread process the number img_idx element of image */
	int img_idx = blockIdx.x * blockSize + threadIdx.x;
	if (img_idx >= (rows * cols)) return;
	int r = img_idx / cols;		 /* row position of the pixel, range [0, rows - 1] */
	int c = img_idx - r * cols;  /* col position of the pixel, range [0, cols - 1] */
	/****************************************************************************
	* Compute the x-derivative. Adjust the derivative at the borders to avoid
	* losing pixels.
	****************************************************************************/
	if (c > 0 && c < cols - 1 /* && cols >= 3 */) d_delta_x[img_idx] = d_smoothedim[img_idx + 1] - d_smoothedim[img_idx - 1];
	else if (c == 0 /* && c + 1 < cols */) d_delta_x[img_idx] = d_smoothedim[img_idx + 1] - d_smoothedim[img_idx];
	else if (c == cols - 1 /* && c - 1 >= 0 */) d_delta_x[img_idx] = d_smoothedim[img_idx] - d_smoothedim[img_idx - 1];
	/****************************************************************************
	* Compute the y-derivative. Adjust the derivative at the borders to avoid
	* losing pixels.
	****************************************************************************/
	if (r > 0 && r < cols - 1) d_delta_y[img_idx] = d_smoothedim[img_idx + cols] - d_smoothedim[img_idx - cols];
	else if (r == 0) d_delta_y[img_idx] = d_smoothedim[img_idx + cols] - d_smoothedim[img_idx];
	else if (r == rows - 1) d_delta_y[img_idx] = d_smoothedim[img_idx] - d_smoothedim[img_idx - cols];
}

/*******************************************************************************
* PROCEDURE: magnitude_x_y_kernel
* PURPOSE: Compute the magnitude of the gradient. This is the square root of
* the sum of the squared derivative values.
* NAME: Vuong Pham-duy
* DATE: 10/11/2016
*******************************************************************************/
__global__ void magnitude_x_y_kernel(int rows, int cols, int blockSize,
	short int *d_magnitude, short int *d_delta_x, short int *d_delta_y)
{
	/* This thread process the number img_idx element of image */
	int img_idx = blockIdx.x * blockSize + threadIdx.x;
	if (img_idx >= (rows * cols)) return;
	int sq1, sq2;
	sq1 = (int)d_delta_x[img_idx] * (int)d_delta_x[img_idx];
	sq2 = (int)d_delta_y[img_idx] * (int)d_delta_y[img_idx];
	d_magnitude[img_idx] = (short)(0.5 + sqrt((float)sq1 + (float)sq2));
}
