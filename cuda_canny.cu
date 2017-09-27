/*******************************************************************************
* --------------------------------------------
* PROGRAM: canny_edge_detector
* PURPOSE: This program is a case study on porting algorithm implemented in C to CUDA 
* The original C code is referenced from canny_edge program implemented by Profs. Mike Heath 
* This program also uses some of the funtions from canny_edge program
*
* NAME: Vuong Pham-Duy
*       Faculty of Computer Science and Technology
*       Ho Chi Minh University of Technology, Viet Nam
*       vuongpd95@gmail.com
*
* DATE: 11/10/2016
*
*******************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MAXBLOCKSIZE 512
#define VERBOSE 1
#define DEBUG 0
#define DEBUGFILE "D:\\Vuong_only\\Images\\edge_out\\debug.txt"
/****************************************************************************
* Declaration of reading and writing pgm image functions, taken directly from 
* C implementation version. There is no need to convert these funcs to CUDA.
****************************************************************************/
int read_pgm_image(char *infilename, unsigned char **image, int *rows, int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows, int cols, char *comment, int maxval);
/****************************************************************************
* Declaration of canny edge detector functions
****************************************************************************/
void gaussian_smooth(int rows, int cols, float sigma, int blockSize, int gridSize,
	short int **d_smoothedim, unsigned char **d_image);

void x_y_calc(int rows, int cols, int blockSize, int gridSize,
	short int **d_delta_x, short int **d_delta_y, short int **d_smoothedim, short int **d_magnitude);

void non_maximal_supp(int rows, int cols, int blockSize, int gridSize,
	short int **d_magnitude, short int **d_delta_x, short int **d_delta_y, unsigned char **d_nms);

void apply_hysteresis(int rows, int cols, int blockSize, int gridSize, float thigh, float tlow,
	unsigned char **h_edge, short int **d_magnitude, unsigned char **d_nms);
/****************************************************************************
* Functions used for debugging
****************************************************************************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);
int write_debug_file(char *outfilename, short int *image, int rows, int cols);
int write_debug_file(char *outfilename, unsigned char *image, int rows, int cols);
/****************************************************************************
* Program start from here
****************************************************************************/
double cuda_canny(char *infilename, float sigma, float tlow, float thigh, double &init_time)
{
	clock_t begin, end,			/* Calculate time to run the main funtion */
		begin_first_malloc,
		begin_first_other;
	double time_spent;			/* Store full program time in here */
	double other_malloc;		/* Store other malloc call time in here */  
	begin = clock();

	char outfilename[128];    /* Name of the output "edge" image */
	unsigned char *h_image, *d_image; /* The input image */ /*h stands for host/memory in CPU */
	unsigned char *h_edge;	/* The output edge image */
	int rows, cols;           /* The dimensions of the image. */

	unsigned char *d_nms,		/* Points that are local maximal magnitude. */
				*h_nms;			/* FOR DEBUGGING */
	short int *d_smoothedim,    /* The image after gaussian smoothing.      */
		*d_delta_x,				/* The first devivative image, x-direction. */
		*d_delta_y,				/* The first derivative image, y-direction. */
		*d_magnitude;			/* The magnitude of the gadient image.  */
	short int *h_smoothedim,	/* FOR DEBUGGING */
		*h_delta_x,				/* FOR DEBUGGING */
		*h_delta_y,				/* FOR DEBUGGING */
		*h_magnitude;			/* FOR DEBUGGING */

	/****************************************************************************
	* Read in the image. This read function allocates memory for the image.
	****************************************************************************/
	if (VERBOSE) printf("\n\nEDGE DETECTOR USING CPU & GPU");
	if (VERBOSE) printf("Reading the image %s.\n", infilename);
	if (read_pgm_image(infilename, &h_image, &rows, &cols) == 0){
		fprintf(stderr, "Error reading the input image, %s.\n", infilename);
		exit(1);
	}
	/****************************************************************************
	* Allocate memory
	****************************************************************************/
	// device image
	begin_first_malloc = clock();
	gpuErrchk(cudaMalloc((void**)&d_image, rows * cols * sizeof(unsigned char)));
	end = clock();
	init_time = 0; /* reset in case next command fail */
	init_time = (double)(end - begin_first_malloc) / CLOCKS_PER_SEC; /* Store init time */
	gpuErrchk(cudaMemcpy((void*)d_image, (void*)h_image, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
	// host edge
	h_edge = (unsigned char*)calloc(rows * cols, sizeof(unsigned char));
	// device non-maximum suppression
	begin_first_other = clock();
	gpuErrchk(cudaMalloc((void**)&d_nms, rows * cols * sizeof(unsigned char)));
	end = clock();
	other_malloc = (double)(end - begin_first_other) / CLOCKS_PER_SEC;
	// device smoothedim, delta_x, delta_y, d_magnitude
	gpuErrchk(cudaMalloc((void**)&d_smoothedim, rows * cols * sizeof(short int)));
	gpuErrchk(cudaMalloc((void**)&d_delta_x, rows * cols * sizeof(short int)));
	gpuErrchk(cudaMalloc((void**)&d_delta_y, rows * cols * sizeof(short int)));
	gpuErrchk(cudaMalloc((void**)&d_magnitude, rows * cols * sizeof(short int)));

	if (DEBUG)
	{
		h_smoothedim = (short int*)calloc(rows * cols, sizeof(short int));
		h_delta_x = (short int*)calloc(rows * cols, sizeof(short int));
		h_delta_y = (short int*)calloc(rows * cols, sizeof(short int));
		h_magnitude = (short int*)calloc(rows * cols, sizeof(short int));
		h_nms = (unsigned char*)calloc(rows * cols, sizeof(unsigned char));
	}
	/****************************************************************************
	* calling kernel parameters
	****************************************************************************/
	int blockSize = MAXBLOCKSIZE;
	int gridSize = (rows * cols + blockSize - 1) / blockSize;
	if (VERBOSE) printf("Calling kernel parameters: blockSize = %d, gridSize = %d\n", blockSize, gridSize);
	/****************************************************************************
	* canny edge detector
	****************************************************************************/
	gaussian_smooth(rows, cols, sigma, blockSize, gridSize, &d_smoothedim, &d_image);
	cudaFree(d_image);
	/****************************************************************************
	* DEBUG BLOCK. IGNORE THIS
	****************************************************************************/
	if (DEBUG)
	{
		gpuErrchk(cudaMemcpy((void*)h_smoothedim, (void*)d_smoothedim, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost));
		int tmp;
		tmp = write_debug_file(DEBUGFILE, h_smoothedim, rows, cols);
		if (tmp == 0) printf("Received error in DEBUG block after gaussian smooth!\n");
	}
	/****************************************************************************/
	x_y_calc(rows, cols, blockSize, gridSize, &d_delta_x, &d_delta_y, &d_smoothedim, &d_magnitude);
	/****************************************************************************
	* DEBUG BLOCK. IGNORE THIS
	****************************************************************************/
	if (DEBUG)
	{
		gpuErrchk(cudaMemcpy((void*)h_magnitude, (void*)d_magnitude, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost));
		int tmp;
		tmp = write_debug_file(DEBUGFILE, h_magnitude, rows, cols);
		if (tmp == 0) printf("Received error in DEBUG block after derrivative_x_y!\n");
	}
	/****************************************************************************/
	cudaFree(d_smoothedim);

	non_maximal_supp(rows, cols, blockSize, gridSize, &d_magnitude, &d_delta_x, &d_delta_y, &d_nms);
	/****************************************************************************
	* DEBUG BLOCK. IGNORE THIS
	****************************************************************************/
	if (DEBUG)
	{
		gpuErrchk(cudaMemcpy((void*)h_nms, (void*)d_nms, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		int tmp;
		tmp = write_debug_file(DEBUGFILE, h_nms, rows, cols);
		if (tmp == 0) printf("Received error in DEBUG block after non-maximal supp!\n");
	}
	/****************************************************************************/
	cudaFree(d_delta_x);
	cudaFree(d_delta_y);

	apply_hysteresis(rows, cols, blockSize, gridSize, thigh, tlow, &h_edge, &d_magnitude, &d_nms);
	cudaFree(d_magnitude);
	cudaFree(d_nms);

	/****************************************************************************
	* Write out the edge image to a file.
	****************************************************************************/
	sprintf_s(outfilename, sizeof(outfilename), "%s_s_%3.2f_l_%3.2f_h_%3.2f_cuda.pgm", infilename,
		sigma, tlow, thigh);
	if (VERBOSE) printf("Writing the edge iname in the file %s.\n", outfilename);
	if (write_pgm_image(outfilename, h_edge, rows, cols, "", 255) == 0){
		fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
		exit(1);
	}
	free(h_image);
	free(h_edge);
	if (DEBUG)
	{
		free(h_nms);
		free(h_smoothedim);
		free(h_delta_x);
		free(h_delta_y);
		free(h_magnitude);
	}
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	if (VERBOSE)
	{
		printf("\nExecution Time of the program with the first cudaMalloc call: %f\n", time_spent);
		printf("Execution Time of the first cudaMalloc call: %f\n", init_time);
		printf("Execution Time of other cudaMalloc call: %f\n", other_malloc);
		printf("Execution Time without the first cudaMalloc call: %f\n", time_spent - init_time);
	}
	return 0;
}
