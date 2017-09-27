/*******************************************************************************
* PROGRAM: canny_edge_detector
* FILE: ui.cu / a small user interface, use WIN32 graphic lib
* PURPOSE: This program is a case study on porting algorithm implemented in C to CUDA
* The original C code is referenced from canny_edge program implemented by Profs. Mike Heath
* This program also uses some of the funtions from canny_edge program
* PURPOSE: Apply Gaussian Smooth to input pgm image
* NAME: Vuong Pham-Duy
*       Faculty of Computer Science and Technology
*       Ho Chi Minh University of Technology, Viet Nam
*       vuongpd95@gmail.com
* DATE: 11/10/2016
*******************************************************************************/
#include<stdio.h>
#include<stdlib.h>

#define INFILENAME "D:\\Vuong_only\\Images\\pgm_in\\img2.pgm"
#define DEBUGFILE "D:\\Vuong_only\\Images\\edge_out\\debug.txt"
#define SIGMA 1.5
#define TLOW 0.35
#define THIGH 0.75
double cuda_canny(char *infilename, float sigma, float tlow, float thigh, double &init_time);
double one_canny(char *infilename, float sigma, float tlow, float thigh);

int main()
{
	double one, cuda, init;
	// TODO UI codes go here
	cuda = cuda_canny(INFILENAME, SIGMA, TLOW, THIGH, init);
	one = one_canny(INFILENAME, SIGMA, TLOW, THIGH);
	printf("\n\nRESULT:\nCUDA: %f INIT_TIME: %f\nCPU: %f \n", cuda, init, one);
	return 0;
}
