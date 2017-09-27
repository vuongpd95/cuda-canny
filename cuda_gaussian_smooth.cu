/*******************************************************************************
* PROGRAM: canny_edge_detector
* FILE: gaussian_smooth.cu 
* PURPOSE: Apply Gaussian Smooth to input pgm image
* NAME: Vuong Pham-Duy
*       Faculty of Computer Science and Technology
*       Ho Chi Minh University of Technology, Viet Nam
*       vuongpd95@gmail.com
* DATE: 11/10/2016
*******************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VERBOSE 1
#define BOOSTBLURFACTOR 90.0
/****************************************************************************
* Functions used for debugging
****************************************************************************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);

void make_gaussian_kernel(float sigma, 
	float **kernel, int *windowsize);

__global__ void gblur_xdir_kernel(int rows, int cols, int blockSize, float sigma, int windowsize,
	float *d_kernel, float *d_tempim, unsigned char *d_image);

__global__ void gblur_ydir_kernel(int rows, int cols, int blockSize, float sigma, int windowsize,
	float *d_kernel, float *d_tempim, short int *d_smoothedim);

/*******************************************************************************
* PROCEDURE: gaussian_smooth
* PURPOSE: Blur an image with a gaussian filter.
* NAME: Vuong Pham-duy
* DATE: 10/11/2016
*******************************************************************************/
void gaussian_smooth(int rows, int cols, float sigma, int blockSize, int gridSize,
	short int **d_smoothedim, unsigned char **d_image)
{
	int windowsize;/* Dimension of the gaussian kernel. */
	float *h_kernel, *d_kernel;/* A one dimensional gaussian kernel in host/device. */
	float *d_tempim;/* Buffer for separable filter gaussian smoothing. */
	/****************************************************************************
	* Create a 1-dimensional gaussian smoothing kernel.
	****************************************************************************/
	if (VERBOSE) printf("Computing the gaussian smoothing kernel.\n");
	make_gaussian_kernel(sigma, &h_kernel, &windowsize);
	/****************************************************************************
	* Allocate memory for kernel, the tmp buffer image and the smoothed image
	****************************************************************************/
	gpuErrchk(cudaMalloc((void**)&d_tempim, rows * cols * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_kernel, windowsize * sizeof(float)));

	gpuErrchk(cudaMemcpy((void*)d_kernel, (void*)h_kernel, windowsize * sizeof(float), cudaMemcpyHostToDevice));

	if (VERBOSE) printf("Smoothing the image using a gaussian kernel.\n");

	gblur_xdir_kernel<<<gridSize, blockSize>>>(rows, cols, blockSize, sigma, windowsize, d_kernel, d_tempim, (*d_image));

	gblur_ydir_kernel<<<gridSize, blockSize>>>(rows, cols, blockSize, sigma, windowsize, d_kernel, d_tempim, (*d_smoothedim));
	
	gpuErrchk(cudaFree(d_kernel));
	gpuErrchk(cudaFree(d_tempim));
	free(h_kernel);
}
__global__ void gblur_xdir_kernel(int rows, int cols, int blockSize, float sigma, int windowsize,
	float *d_kernel, float *d_tempim, unsigned char *d_image)
{
	/* This thread process the number img_idx element of image */
	int img_idx = blockIdx.x * blockSize + threadIdx.x;
	if (img_idx >= (rows * cols)) return;
	int r = img_idx / cols;	/* row position of the pixel, range [0, rows - 1] */
	int c = img_idx - r * cols;/* col position of the pixel, range [0, cols - 1] */
	int center = windowsize / 2;/* Half of the windowsize. */
	/****************************************************************************
	* Gaussian smooth in x direction
	****************************************************************************/
	int counter; /* gaussian kernel counter */
	float dot = 0.0;
	float sum = 0.0;
	for (counter = (-center); counter <= center; counter++){
		if (((c + counter) >= 0) && ((c + counter) < cols)){
			dot += (float)d_image[img_idx + counter] * d_kernel[center + counter];
			sum += d_kernel[center + counter];
		}
	}
	d_tempim[img_idx] = dot / sum;
}
__global__ void gblur_ydir_kernel(int rows, int cols, int blockSize, float sigma, int windowsize, 
	float *d_kernel, float *d_tempim, short int *d_smoothedim)
{
	// This thread process the number img_idx element of image
	int img_idx = blockIdx.x * blockSize + threadIdx.x;
	if (img_idx >= (rows * cols)) return;
	int r = img_idx / cols;/* row position of the pixel, range [0, rows - 1] */
	int c = img_idx - r * cols;/* col position of the pixel, range [0, cols - 1] */
	int center = windowsize / 2;/* Half of the windowsize. */
	/****************************************************************************
	* Gaussian smooth in y direction
	****************************************************************************/
	int rr;
	float dot = 0.0;
	float sum = 0.0;
	for (rr = (-center); rr <= center; rr++){
		if (((r + rr) >= 0) && ((r + rr) < rows)){
			dot += d_tempim[(r + rr)*cols + c] * d_kernel[center + rr];
			sum += d_kernel[center + rr];
		}
	}
	d_smoothedim[img_idx] = (short int)(dot*BOOSTBLURFACTOR / sum + 0.5);
}

/*******************************************************************************
* PROCEDURE: make_gaussian_kernel
* PURPOSE: Create a one dimensional gaussian kernel.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
	int i, center;
	float x, fx, sum = 0.0;

	*windowsize = 1 + 2 * ceil(2.5 * sigma);
	center = (*windowsize) / 2;

	if (VERBOSE) printf("      The kernel has %d elements.\n", *windowsize);
	if ((*kernel = (float *)calloc((*windowsize), sizeof(float))) == NULL){
		fprintf(stderr, "Error callocing the gaussian kernel array.\n");
		exit(1);
	}

	for (i = 0; i<(*windowsize); i++){
		x = (float)(i - center);
		fx = pow(2.71828, -0.5*x*x / (sigma*sigma)) / (sigma * sqrt(6.2831853));
		(*kernel)[i] = fx;
		sum += fx;
	}

	for (i = 0; i<(*windowsize); i++) (*kernel)[i] /= sum;

	if (VERBOSE){
		printf("The filter coefficients are:\n");
		for (i = 0; i<(*windowsize); i++)
			printf("kernel[%d] = %f\n", i, (*kernel)[i]);
	}
}
