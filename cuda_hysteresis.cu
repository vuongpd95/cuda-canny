/*******************************************************************************
* PROGRAM: canny_edge_detector
* FILE: non_maximal_supp.cu 
* PURPOSE: Apply maximal suppression
* NAME: Vuong Pham-Duy
*       Faculty of Computer Science and Technology
*       Ho Chi Minh University of Technology, Viet Nam
*       vuongpd95@gmail.com
* DATE: 11/10/2016
*******************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_32_atomic_functions.h"

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0
#define HIST_SIZE 32768
#define VERBOSE 1

__global__ void hist_calc_kernel(int rows, int cols, int blockSize,
	short int *d_magnitude, unsigned char *d_nms, unsigned char *d_edge, int *d_hist);

__global__ void hc_calc_kernel(int blockSize, float thigh, 
	float *d_highcount, int *d_max_mag, int *d_hist);

__global__ void thresholding_kernel(int rows, int cols, int blockSize, int highthreshold, int lowthreshold, 
	short int *d_magnitude, unsigned char *d_edge);

__global__ void edge_supp_kernel(int rows, int cols, int blockSize, 
	unsigned char *d_edge);

void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,
	int cols);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);

/*******************************************************************************
* PROCEDURE: apply_hysteresis
* PURPOSE: apply hysteresis
* NAME: Vuong Pham-duy
* DATE: 10/11/2016
*******************************************************************************/
void apply_hysteresis(int rows, int cols, int blockSize, int gridSize, float thigh, float tlow,
	unsigned char **h_edge, short int **d_magnitude, unsigned char **d_nms)
{
	int *h_hist, *d_hist;/* histogram in host/device */
	unsigned char *d_edge;/* edge image in device */
	float *h_highcount, *d_highcount;/* highcount value in host/device */
	int *h_max_mag, *d_max_mag, /* maximum magnitude in host/device*/
		hcount, /* integer value of highcount */ 
		r, numedges;/* counter variables */
	int highthreshold, lowthreshold; /* high and low threshold used to apply hysteresis */
	short int *h_magnitude; /* magnitude in host */
	/****************************************************************************
	* Allocate memory
	****************************************************************************/
	h_hist = (int*)calloc(HIST_SIZE, sizeof(int));
	h_highcount = (float*)calloc(1, sizeof(float));
	h_max_mag = (int*)calloc(1, sizeof(int));
	h_magnitude = (short int*)calloc(rows * cols, sizeof(short int));

	gpuErrchk(cudaMalloc((void**)&d_hist, HIST_SIZE * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_edge, rows * cols * sizeof(unsigned char)));
	gpuErrchk(cudaMalloc((void**)&d_highcount, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_max_mag, sizeof(int)));
	/****************************************************************************
	* Initialize memory, if needed
	****************************************************************************/
	gpuErrchk(cudaMemset(d_hist, 0, HIST_SIZE * sizeof(int)));
	gpuErrchk(cudaMemset(d_edge, (unsigned char)NOEDGE, rows * cols * sizeof(unsigned char)));
	gpuErrchk(cudaMemset(d_highcount, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_max_mag, 0, sizeof(int)));
	/****************************************************************************
	* Use hysteresis to mark the edge pixels.
	****************************************************************************/
	if (VERBOSE) printf("Doing hysteresis thresholding.\n");
	hist_calc_kernel<<<gridSize, blockSize>>>(rows, cols, blockSize, (*d_magnitude), (*d_nms), d_edge, d_hist);

	gpuErrchk(cudaMemcpy((void*)h_hist, (void*)d_hist, HIST_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

	/****************************************************************************
	* Calculate lowthreshold and highthreshold
	****************************************************************************/
	// Only need 32768 threads so choose another gridSize
	int gSize = (HIST_SIZE + blockSize - 1)/blockSize;
	hc_calc_kernel<<<gSize, blockSize>>>(blockSize, thigh, d_highcount, d_max_mag, d_hist);
	gpuErrchk(cudaMemcpy((void*)h_highcount, (void*)d_highcount, sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy((void*)h_max_mag, (void*)d_max_mag, sizeof(int), cudaMemcpyDeviceToHost));

	// calc highthreshold and lowthreshold
	hcount = (int)(*h_highcount + 0.5);
	r = 1;
	numedges = h_hist[1];
	while ((r<((*h_max_mag) - 1)) && (numedges < hcount)){
		r++;
		numedges += h_hist[r];
	}
	highthreshold = r;
	lowthreshold = (int)(highthreshold * tlow + 0.5);
	if (VERBOSE){
		printf("The input low and high fractions of %f and %f computed to\n",
			tlow, thigh);
		printf("magnitude of the gradient threshold values of: %d %d\n",
			lowthreshold, highthreshold);
	}

	thresholding_kernel<<<gridSize, blockSize>>>(rows, cols, blockSize, highthreshold, lowthreshold, (*d_magnitude), d_edge);
	gpuErrchk(cudaMemcpy((void*)(*h_edge), (void*)d_edge, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy((void*)h_magnitude, (void*)(*d_magnitude), rows * cols * sizeof(short int), cudaMemcpyDeviceToHost));
	/****************************************************************************
	* This loop looks for pixels above the highthreshold to locate edges and
	* then calls follow_edges to continue the edge.
	****************************************************************************/
	int c, pos;
	for (r = 0, pos = 0; r<rows; r++){
		for (c = 0; c<cols; c++, pos++){
			if (((*h_edge)[pos] == EDGE) && (h_magnitude[pos] >= highthreshold)){
				follow_edges((*h_edge + pos), (h_magnitude + pos), lowthreshold, cols);
			}
		}
	}
	/****************************************************************************
	* Set all the remaining possible edges to non-edges.
	****************************************************************************/
	gpuErrchk(cudaMemcpy((void*)d_edge, (void*)(*h_edge), rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
	edge_supp_kernel<<<gridSize, blockSize>>>(rows, cols, blockSize, d_edge);

	gpuErrchk(cudaMemcpy((void*)(*h_edge), (void*)d_edge, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	/* Free unneeded memory */
	cudaFree(d_highcount);
	cudaFree(d_max_mag);
	cudaFree(d_hist);
	cudaFree(d_edge);

	free(h_highcount);
	free(h_max_mag);
	free(h_hist);
	free(h_magnitude);
}

__global__ void hist_calc_kernel(int rows, int cols, int blockSize,
	short int *d_magnitude, unsigned char *d_nms, unsigned char *d_edge, int *d_hist)
{
	/* This thread process the number img_idx element of image */
	int img_idx = blockIdx.x * blockSize + threadIdx.x;
	if (img_idx >= (rows * cols)) return;
	if (d_nms[img_idx] == POSSIBLE_EDGE)
	{
		d_edge[img_idx] = POSSIBLE_EDGE;
		atomicAdd(&d_hist[d_magnitude[img_idx]], 1);
	}
}

__global__ void hc_calc_kernel(int blockSize, float thigh, 
	float *d_highcount, int *d_max_mag, int *d_hist)
{
	/* This thread process the number img_idx element of image */
	int hist_idx = blockIdx.x * blockSize + threadIdx.x;
	if (hist_idx >= HIST_SIZE || hist_idx == 1) return;

	int hist_tmp = d_hist[hist_idx];
	if (hist_tmp != 0) atomicExch(d_max_mag, hist_idx);
	atomicAdd(d_highcount, (float) hist_tmp * thigh);
}

__global__ void thresholding_kernel(int rows, int cols, int blockSize, int highthreshold, int lowthreshold, 
	short int *d_magnitude, unsigned char *d_edge)
{
	/* This thread process the number img_idx element of image */
	int img_idx = blockIdx.x * blockSize + threadIdx.x;
	if (img_idx >= (rows * cols)) return;
	if ((d_edge[img_idx] == POSSIBLE_EDGE) && (d_magnitude[img_idx] >= highthreshold)){
		d_edge[img_idx] = EDGE;
	}
}

__global__ void edge_supp_kernel(int rows, int cols, int blockSize, 
	unsigned char *d_edge)
{
	/* This thread process the number img_idx element of image */
	int img_idx = blockIdx.x * blockSize + threadIdx.x;
	if (img_idx >= (rows * cols)) return;
	if (d_edge[img_idx] != EDGE) d_edge[img_idx] = NOEDGE;
}
/*******************************************************************************
* PROCEDURE: follow_edges
* PURPOSE: This procedure edges is a recursive routine that traces edgs along
* all paths whose magnitude values remain above some specifyable lower
* threshhold.
* NAME: Mike Heath
* DATE: 2/15/96
*******************************************************************************/
void follow_edges(unsigned char *edgemapptr, short *edgemagptr, short lowval,
	int cols)
{
	// TODO convert to CUDA
	short *tempmagptr;
	unsigned char *tempmapptr;
	int i;
	// float thethresh;
	int x[8] = { 1, 1, 0, -1, -1, -1, 0, 1 },
		y[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };

	for (i = 0; i<8; i++){
		tempmapptr = edgemapptr - y[i] * cols + x[i];
		tempmagptr = edgemagptr - y[i] * cols + x[i];

		if ((*tempmapptr == POSSIBLE_EDGE) && (*tempmagptr > lowval)){
			*tempmapptr = (unsigned char)EDGE;
			follow_edges(tempmapptr, tempmagptr, lowval, cols);
		}
	}
}
