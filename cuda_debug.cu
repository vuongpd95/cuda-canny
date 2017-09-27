/*******************************************************************************
* FILE: debug.cu
* PURPOSE: Contain functions used to keep track of CUDA functions errors.
* NAME: Vuong Pham-Duy
*       Faculty of Computer Science and Technology
*       Ho Chi Minh University of Technology, Viet Nam
*       vuongpd95@gmail.com
* DATE: 11/10/2016
*******************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

/******************************************************************************
* Macro: gpuErrchk
* Purpose: a simple wrap macro used to detect gpu errors.
******************************************************************************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
/******************************************************************************
* Function: write_debug_file
* Purpose: These function write an array of unsigned char or short int to a file
* whose its absolute path is defined by DEBUGFILE. "debug.txt" will show us some
* infomation about the errors that happened.
******************************************************************************/
int write_debug_file(char *outfilename, short int *image, int rows, int cols)
{
	FILE *fp;
	errno_t err;
	/***************************************************************************
	* Open the output image file for writing if a filename was given. If no
	* filename was provided, set fp to write to standard output.
	***************************************************************************/
	if (outfilename == NULL) fp = stdout;
	else{
		if ((err = fopen_s(&fp, outfilename, "w")) != 0){
			fprintf(stderr, "Error writing the file %s in write2file(): %d \n",
				outfilename, err);
			return(0);
		}
	}
	/***************************************************************************
	* Write the image data to the file.
	***************************************************************************/
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++) fprintf(fp, "%10d|", image[i * cols + j]);
		fprintf(fp, "\n");
	}
	printf("Finish writting %d lines and %d cols to debug.txt\n", i, j);
	if (fp != stdout) fclose(fp);
	return(1);
}
int write_debug_file(char *outfilename, unsigned char *image, int rows, int cols)
{
	FILE *fp;
	errno_t err;
	/***************************************************************************
	* Open the output image file for writing if a filename was given. If no
	* filename was provided, set fp to write to standard output.
	***************************************************************************/
	if (outfilename == NULL) fp = stdout;
	else{
		if ((err = fopen_s(&fp, outfilename, "w")) != 0){
			fprintf(stderr, "Error writing the file %s in write2file(): %d \n",
				outfilename, err);
			return(0);
		}
	}
	/***************************************************************************
	* Write the image data to the file.
	***************************************************************************/
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++) fprintf(fp, "%3u|", image[i * cols + j]);
		fprintf(fp, "\n");
	}
	printf("Finish writting %d lines to debug.txt\n", i);
	if (fp != stdout) fclose(fp);
	return(1);
}
