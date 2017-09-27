/*******************************************************************************
* PROGRAM: canny_edge_detector
* FILE: non_maximal_supp.cu 
* PURPOSE: Apply non maximal suppression.
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

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0
#define VERBOSE 1

__global__ void non_max_supp_kernel(int rows, int cols, int blockSize,
	short int *d_magnitude, short int *d_delta_x, short int *d_delta_y, unsigned char *d_nms);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);
/*******************************************************************************
* PROCEDURE: non_maximal_supp
* PURPOSE: perform non-maximal suppression
* NAME: Vuong Pham-duy
* DATE: 10/11/2016
*******************************************************************************/
void non_maximal_supp(int rows, int cols, int blockSize, int gridSize,
	short int **d_magnitude, short int **d_delta_x, short int **d_delta_y, unsigned char **d_nms)
{
	/****************************************************************************
	* Perform non-maximal suppression.
	****************************************************************************/
	if (VERBOSE) printf("Doing the non-maximal suppression.\n");
	gpuErrchk(cudaMemset((*d_nms), 0, rows * cols * sizeof(unsigned char)));
	non_max_supp_kernel<<<gridSize, blockSize>>>(rows, cols, blockSize, (*d_magnitude), (*d_delta_x), (*d_delta_y), (*d_nms));
}
/*******************************************************************************
* PROCEDURE: non_max_supp_kernel
* PURPOSE: This routine applies non-maximal suppression to the magnitude of
* every pixel of the gradient image
* NAME: Vuong Pham-duy
* DATE: 10/11/2016
*******************************************************************************/
__global__ void non_max_supp_kernel(int rows, int cols, int blockSize,
	short int *d_magnitude, short int *d_delta_x, short int *d_delta_y, unsigned char *d_nms)
{
	/* This thread process the number img_idx element of image */
	int img_idx = blockIdx.x * blockSize + threadIdx.x;
	if (img_idx >= (rows * cols)) return;
	int r = img_idx / cols;		 /* row position of the pixel, range [0, rows - 1] */
	int c = img_idx - r * cols;  /* col position of the pixel, range [0, cols - 1] */
	if ((r != rows - 1) && (r != 0) && (c != 0) && (c != cols - 1))
	{
		short m00;
		m00 = d_magnitude[img_idx];
		if (m00 == 0) d_nms[img_idx] = (unsigned char)NOEDGE;
		else
		{
			short gx, gy, z1, z2;
			float mag1, mag2, xperp, yperp;
			gx = d_delta_x[img_idx];
			gy = d_delta_y[img_idx];
			xperp = -gx / ((float)m00);
			yperp = gy / ((float)m00);
			if (gx >= 0){
				if (gy >= 0){
					if (gx >= gy)
					{
						/* 111 */
						/* Left point */
						z1 = d_magnitude[img_idx - 1]; 
						z2 = d_magnitude[img_idx - cols - 1];

						mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;

						/* Right point */
						z1 = d_magnitude[img_idx + 1];
						z2 = d_magnitude[img_idx + cols + 1];

						mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
					}
					else
					{
						/* 110 */
						/* Left point */
						z1 = d_magnitude[img_idx - cols];
						z2 = d_magnitude[img_idx - cols - 1];

						mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;

						/* Right point */
						z1 = d_magnitude[img_idx + cols];
						z2 = d_magnitude[img_idx + cols + 1];

						mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp;
					}
				}
				else
				{
					if (gx >= -gy)
					{
						/* 101 */
						/* Left point */
						z1 = d_magnitude[img_idx - 1];
						z2 = d_magnitude[img_idx + cols - 1];

						mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;

						/* Right point */
						z1 = d_magnitude[img_idx + 1];
						z2 = d_magnitude[img_idx - cols + 1];

						mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
					}
					else
					{
						/* 100 */
						/* Left point */
						z1 = d_magnitude[img_idx + cols];
						z2 = d_magnitude[img_idx + cols - 1];

						mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;

						/* Right point */
						z1 = d_magnitude[img_idx - cols];
						z2 = d_magnitude[img_idx - cols + 1];

						mag2 = (z1 - z2)*xperp + (m00 - z1)*yperp;
					}
				}
			}
			else
			{
				if (gy >= 0)
				{
					if (-gx >= gy)
					{
						/* 011 */
						/* Left point */
						z1 = d_magnitude[img_idx + 1];
						z2 = d_magnitude[img_idx - cols + 1];

						mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;

						/* Right point */
						z1 = d_magnitude[img_idx - 1];
						z2 = d_magnitude[img_idx + cols - 1];

						mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
					}
					else
					{
						/* 010 */
						/* Left point */
						z1 = d_magnitude[img_idx - cols];
						z2 = d_magnitude[img_idx - cols + 1];

						mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;

						/* Right point */
						z1 = d_magnitude[img_idx + cols];
						z2 = d_magnitude[img_idx + cols - 1];

						mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
					}
				}
				else
				{
					if (-gx > -gy)
					{
						/* 001 */
						/* Left point */
						z1 = d_magnitude[img_idx + 1];
						z2 = d_magnitude[img_idx + cols + 1];

						mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;

						/* Right point */
						z1 = d_magnitude[img_idx - 1];
						z2 = d_magnitude[img_idx - cols - 1];

						mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
					}
					else
					{
						/* 000 */
						/* Left point */
						z1 = d_magnitude[img_idx + cols];
						z2 = d_magnitude[img_idx + cols + 1];

						mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;

						/* Right point */
						z1 = d_magnitude[img_idx - cols];
						z2 = d_magnitude[img_idx - cols - 1];

						mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
					}
				}
			}

			/* Now determine if the current point is a maximum point */

			if ((mag1 > 0.0) || (mag2 > 0.0))
			{
				d_nms[img_idx] = (unsigned char)NOEDGE;
			}
			else
			{
				if (mag2 == 0.0)
					d_nms[img_idx] = (unsigned char)NOEDGE;
				else
					d_nms[img_idx] = (unsigned char)POSSIBLE_EDGE;
			}
		}
	}
}
