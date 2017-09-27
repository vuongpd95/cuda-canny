# cuda-canny

This is my first attempt in using CUDA framework to write an edge detection software. The software is based on Canny Edge Detector Mechanism.

Instead of writting the detector from scratch, I ported the Canny Edge Detector, which was written by Profs. Mike Heath, to CUDA language. Every functions in the source code written by Profs. Mike Heath is indicated with comments.

The CUDA Canny Edge Detector yielded several times better execution time than its counter C version. The detector can still be improved in many areas since my experience in writting CUDA codes is still of a beginner at the time.

The UI of the software haven't done yet since I decided to drop the project. Modify INFILENAME (ui.cu) to your own image source path to test the detector. To build the source code, please do a little research to find a nvcc build command. You might want to install CUDA 8.0 because it was the version that I used to write the code.
