#include<cuda.h>
#include<cuda_runtime.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include<time.h>
#define _USE_MATH_DEFINES
#include <cmath>
#define PI 3.14159265358979323846
#define frame_size 30
#define gaussianframe_size 28
#define convolution_mask_size 3
#define block_size (frame_size + convolution_mask_size - 1)
#define gaussian_mask_size 5
#define gaussian_block_size (gaussianframe_size + gaussian_mask_size - 1)
__constant__ float convolutionGaussianxStore[256];
__constant__ float convolutionsobelxStore[256];
__constant__ float convolutionsobelyStore[256];
//
bool seq=false;
template <int BLOCK_SIZE> __global__ void difference_filter(unsigned char *output, unsigned char *edges1, unsigned char *edges2, int width1, int height1, int threshold1) {
	
	int height = height1;
	int width = width1;
	int threshold = threshold1;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int i = r * width + c;

	// Set it to 0 initially
	unsigned char res = (unsigned char)0;
	int cropSize = 7;
	if (r > cropSize && c > cropSize && r < height - cropSize && c < width - cropSize && edges1[i] != edges2[i]) {
		// Set to 255 if there is a mismatch in pixel 
		res = (unsigned char)255;
		for (int m = -threshold; m <= threshold; m++) {
			for (int n = -threshold; n <= threshold; n++) {
				if (c + m > 0 && r + n > 0 && c + m < width && r + n < height) {
					if (edges1[(r + n) * width + c + m] == edges2[i]) {
						res = (unsigned char)0;
					}
				}
			}
		}
		output[i] = res;
	}
}

void serial_difference_filter(unsigned char *difference, unsigned char *edges1, unsigned char *edges2, int width, int height, int threshold) {
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			difference[y * width + x] = 0;
			if (edges1[y * width + x] != edges2[y * width + x]) {
				difference[y * width + x] = (unsigned char)255;
				for (int m = -threshold; m <= threshold; m++) {
					for (int n = -threshold; n <= threshold; n++) {
						if (x + m > 0 && y + n > 0 && x + m < width && y + n < height) {
							if (edges1[(y + n) * width + x + m] == edges2[y * width + x]) {
								difference[y * width + x] =(unsigned char) 0;
							}
						}
					}
				}
			}
		}
	}
}
void serial_thresholding_and_suppression(unsigned char *output, int input_width, int input_height, unsigned char *g_x, unsigned char  *g_y, int high_threshold, int low_threshold) {
	for (int r = 0; r < input_height; r++) {
		for (int c = 0; c < input_width; c++) {
			int i = r * input_width + c;
			// First, initialize the current pixel to zero (non-edge)
			output[i] = (unsigned char)0;
			// Boundary conditions
			if (r > 1 && c > 1 && r < input_height - 1 && c < input_width - 1) {
				double magnitude = sqrt(pow((double)g_x[i], 2) + pow((double)g_y[i], 2));
				if (magnitude > high_threshold) {
					// Non-maximum suppression: determine magnitudes in the surrounding pixel and the gradient direction of the current pixel
					double magnitude_above = sqrt(pow((double)g_x[(r - 1) * input_width + c], 2) + pow((double)g_y[(r - 1) * input_width + c], 2));
					double magnitude_below = sqrt(pow((double)g_x[(r + 1) * input_width + c], 2) + pow((double)g_y[(r + 1) * input_width + c], 2));
					double magnitude_left = sqrt(pow((double)g_x[r * input_width + c - 1], 2) + pow((double)g_y[r * input_width + c - 1], 2));
					double magnitude_right = sqrt(pow((double)g_x[r * input_width + c + 1], 2) + pow((double)g_y[r * input_width + c + 1], 2));
					double magnitude_upper_right = sqrt(pow((double)g_x[(r + 1) * input_width + c + 1], 2) + pow((double)g_y[(r + 1) * input_width + c + 1], 2));
					double magnitude_upper_left = sqrt(pow((double)g_x[(r + 1) * input_width + c - 1], 2) + pow((double)g_y[(r + 1) * input_width + c - 1], 2));
					double magnitude_lower_right = sqrt(pow((double)g_x[(r - 1) * input_width + c + 1], 2) + pow((double)g_y[(r - 1) * input_width + c + 1], 2));
					double magnitude_lower_left = sqrt(pow((double)g_x[(r - 1) * input_width + c - 1], 2) + pow((double)g_y[(r - 1) * input_width + c - 1], 2));
					double theta = atan2((double)g_y[i], (double)g_x[i]);

					// Check if the current pixel is a ridge pixel, e.g. maximized in the gradient direction
					int verticalCheck = (PI / 3.0 < theta && theta < 2.0*PI / 3.0) || (-2.0*PI / 3.0 < theta && theta < -PI / 3.0);
					int isVerticalMax = verticalCheck && magnitude > magnitude_below && magnitude > magnitude_above;
					int horizontalCheck = (-PI / 6.0 < theta && theta < PI / 6.0) || (-PI < theta && theta < -5.0*PI / 6.0) || (5 * PI / 6.0 < theta && theta < PI);
					int isHorizontalMax = horizontalCheck && magnitude > magnitude_right && magnitude > magnitude_left;
					int positiveDiagonalCheck = (theta > PI / 6.0 && theta < PI / 3.0) || (theta < -2.0*PI / 3.0 && theta > -5.0*PI / 6.0);
					int isPositiveDiagonalMax = positiveDiagonalCheck && magnitude > magnitude_upper_right && magnitude > magnitude_lower_left;
					int negativeDiagonalCheck = (theta > 2.0*PI / 3.0 && theta < 5.0*PI / 6.0) || (theta < -PI / 6.0 && theta > -PI / 3.0);
					int isNegativeDiagonalMax = negativeDiagonalCheck && magnitude > magnitude_lower_right && magnitude > magnitude_upper_left;

					// Consider a surrounding apron around the current pixel to catch potentially disconnected pixel nodes
					int maskSize = 2;
					if (isVerticalMax || isHorizontalMax || isPositiveDiagonalMax || isNegativeDiagonalMax) {
						output[i] = (unsigned char)255;
						for (int m = -maskSize; m <= maskSize; m++) {
							for (int n = -maskSize; n <= maskSize; n++) {
								if (r + m > 0 && r + m < input_height && c + n > 0 && c + n < input_width) {
									if (sqrt(pow((double)g_x[(r + m) * input_width + c + n], 2) + pow((double)g_y[(r + m) * input_width + c + n], 2)) > low_threshold) {
										output[(r + m) * input_width + c + n] = (unsigned char)255;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}



template <int BLOCK_SIZE> __global__ void thresholding_and_suppression(unsigned char *output, unsigned char *magnitude, unsigned char *angle, int input_width1, int input_height1, int high_threshold1, int low_threshold1) {
	int input_height = input_height1;
	int high_threshold= high_threshold1;
	int input_width =  input_width1;
	int low_threshold =  low_threshold1;
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = r * (input_width)+c;
	//testcode
	const int tx = threadIdx.x;                           // --- Local thread x index
	const int ty = threadIdx.y;                           // --- Local thread y index

	const int tx_g = blockIdx.x * blockDim.x + tx;        // --- Global thread x index
	const int ty_g = blockIdx.y * blockDim.y + ty;        // --- Global thread y index
	int BLOCK_WIDTH = 30;
	int BLOCK_HEIGHT = 30;
	__shared__ float sharedmem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	// --- Fill the shared memory border with zeros
	if (tx == 0)                      sharedmem[tx][ty + 1] = 0;    // --- left border
	else if (tx == BLOCK_WIDTH - 1)     sharedmem[tx + 2][ty + 1] = 0;    // --- right border
	if (ty == 0) {
		sharedmem[tx + 1][ty] = 0;    // --- upper border
		if (tx == 0)                  sharedmem[tx][ty] = 0;    // --- top-left corner
		else if (tx == BLOCK_WIDTH - 1) sharedmem[tx + 2][ty] = 0;    // --- top-right corner
	}
	else if (ty == BLOCK_HEIGHT - 1) {
		sharedmem[tx + 1][ty + 2] = 0;    // --- bottom border
		if (tx == 0)                  sharedmem[tx][ty + 2] = 0;    // --- bottom-left corder
		else if (tx == BLOCK_WIDTH - 1) sharedmem[tx + 2][ty + 2] = 0;    // --- bottom-right corner
	}

	// --- Fill shared memory
	sharedmem[tx + 1][ty + 1] = (double)magnitude[ty_g*input_width + tx_g];      // --- center
	if ((tx == 0) && ((tx_g > 0)))                                      sharedmem[tx][ty + 1] = (double)magnitude[ty_g*input_width + tx_g - 1];      // --- left border
	else if ((tx == BLOCK_WIDTH - 1) && (tx_g < input_width - 1))         sharedmem[tx + 2][ty + 1] = (double)magnitude[ty_g*input_width + tx_g + 1];      // --- right border
	if ((ty == 0) && (ty_g > 0)) {
		sharedmem[tx + 1][ty] = (double)magnitude[(ty_g - 1)*input_width + tx_g];    // --- upper border
		if ((tx == 0) && ((tx_g > 0)))                                  sharedmem[tx][ty] = (double)magnitude[(ty_g - 1)*input_width + tx_g - 1];  // --- top-left corner
		else if ((tx == BLOCK_WIDTH - 1) && (tx_g < input_width - 1))     sharedmem[tx + 2][ty] = (double)magnitude[(ty_g - 1)*input_width + tx_g + 1];  // --- top-right corner
	}
	else if ((ty == BLOCK_HEIGHT - 1) && (ty_g < input_height - 1)) {
		sharedmem[tx + 1][ty + 2] = (double)magnitude[(ty_g + 1)*input_width + tx_g];    // --- bottom border
		if ((tx == 0) && ((tx_g > 0)))                                 sharedmem[tx][ty + 2] = (double)magnitude[(ty_g - 1)*input_width + tx_g - 1];  // --- bottom-left corder
		else if ((tx == BLOCK_WIDTH - 1) && (tx_g < input_width - 1))     sharedmem[tx + 2][ty + 2] = (double)magnitude[(ty_g + 1)*input_width + tx_g + 1];  // --- bottom-right corner
	}
	__syncthreads();

	//testcode
	// First, initialize the current pixel to zero (non-edge)
	output[i] = (unsigned char)0;
	// Boundary conditions
	if (r > 1 && c > 1 && r < input_height - 1 && c < input_width - 1) {
		double magnitude = double(sharedmem[tx+1][ty + 1]);
		if (magnitude > high_threshold) {
			double magnitude_above =(double) sharedmem[tx][ty+1];
			double magnitude_below = (double)sharedmem[tx+2][ty+1];;
			double magnitude_left = (double)sharedmem[tx+1][ty];
			double magnitude_right = (double)sharedmem[tx+1][ty+2];
			double magnitude_upper_right = (double)sharedmem[tx +2][ty+2];
			double magnitude_upper_left = (double)sharedmem[tx +2][ty];
			double magnitude_lower_right = (double)sharedmem[tx ][ty+2];
			double magnitude_lower_left = (double)sharedmem[tx ][ty];


			double theta = double(angle[i]);

			// Check if the current pixel is a ridge pixel, e.g. maximized in the gradient direction
			int verticalCheck = (PI / 3.0 < theta && theta < 2.0*PI / 3.0) || (-2.0*PI / 3.0 < theta && theta < -PI / 3.0);
			int isVerticalMax = verticalCheck && magnitude > magnitude_below && magnitude > magnitude_above;
			int horizontalCheck = (-PI / 6.0 < theta && theta < PI / 6.0) || (-PI < theta && theta < -5.0*PI / 6.0) || (5 * PI / 6.0 < theta && theta < PI);
			int isHorizontalMax = horizontalCheck && magnitude > magnitude_right && magnitude > magnitude_left;
			int positiveDiagonalCheck = (theta > PI / 6.0 && theta < PI / 3.0) || (theta < -2.0*PI / 3.0 && theta > -5.0*PI / 6.0);
			int isPositiveDiagonalMax = positiveDiagonalCheck && magnitude > magnitude_upper_right && magnitude > magnitude_lower_left;
			int negativeDiagonalCheck = (theta > 2.0*PI / 3.0 && theta < 5.0*PI / 6.0) || (theta < -PI / 6.0 && theta > -PI / 3.0);
			int isNegativeDiagonalMax = negativeDiagonalCheck && magnitude > magnitude_lower_right && magnitude > magnitude_upper_left;

			// Consider a surrounding apron around the current pixel to catch potentially disconnected pixel nodes
			int maskSize = 1;
			if (isVerticalMax || isHorizontalMax || isPositiveDiagonalMax || isNegativeDiagonalMax) {
				output[i] = (unsigned char)255;
				for (int m = -maskSize; m <= maskSize; m++) {
					for (int n = -maskSize; n <= maskSize; n++) {
						if (r + m > 0 && r + m < input_height && c + n > 0 && c + n < input_width) {
							if (double(sharedmem[(1+tx + m)] [1+ ty + n]) > low_threshold) {
								output[(r + m) * input_width + c + n] = (unsigned char)255;
							}
						}
					}
				}
			}
		}
	}
}



template<int BlOCK_SIZE,int TILE_SIZE>__global__ void ConvolutionKernel(unsigned char *source , int source_height1, int source_width1,int KERNEL_SIZE1,unsigned char * blurredDataDevice,int flag1)
{
	int source_height = source_height1;
	int source_width = source_width1;
	int KERNEL_SIZE =  KERNEL_SIZE1;
	int flag =  flag1;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;
	int n = KERNEL_SIZE / 2;
	int row_i = row_o - n;
	int col_i = col_o - n;
	__shared__ float N_s[BlOCK_SIZE][BlOCK_SIZE];

	if ((row_i >= 0) && (row_i < source_height) &&
		(col_i >= 0) && (col_i < source_width)) {
		N_s[ty][tx] = float(source[row_i*source_width + col_i]);
	}
	else {
		N_s[ty][tx] = 0.0f;
	}
	__syncthreads();

	float output = 0.0f;

	if (ty < TILE_SIZE && tx < TILE_SIZE) {
		for (int i = 0; i < KERNEL_SIZE; i++) {
			for (int j = 0; j < KERNEL_SIZE; j++) {
				switch (flag)
				{
				case 1:
					output += convolutionGaussianxStore[i*KERNEL_SIZE + j] * N_s[i + ty][j + tx];
					break;
				case 2:
					output += convolutionsobelxStore[i*KERNEL_SIZE + j] * N_s[i + ty][j + tx];
					break;
				case 3:
					output += convolutionsobelyStore[i*KERNEL_SIZE + j] * N_s[i + ty][j + tx];
					break;
				}
					
			}
		}
		//__syncthreads();

		if (row_o < source_height && col_o < source_width)
			blurredDataDevice[row_o * source_width + col_o] = (unsigned char)output;
	}
}

// converts the pythagoran theorem along a vector on the GPU
template<int BlOCK_SIZE>__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c, unsigned char *d)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float base = float(a[idx]);
	float height = float(b[idx]);

	c[idx] = (unsigned char)sqrtf(base*base + height * height);
	d[idx] = (unsigned char)atan2(height, base);
}

// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr)
{
	unsigned char *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}



int main(int argc, char** argv)
{
	// Open a webcamera
	//cv::VideoCapture camera(0);
	//modify 
	cv::VideoCapture camera("./bm_clip_mp4High/img%04d.png");
	//modify 
	cv::Mat          frame,frame2;
	if (!camera.isOpened())
		return -1;
	float totaltime=0;
	// Create the capture windows
	//cv::namedWindow("Source");
	//cv::namedWindow("Greyscale");
	//cv::namedWindow("Blurred");
	//cv::namedWindow("Sobel");

	// Create the cuda event timers 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Create the gaussian kernel (sum = 159)
	/*const float gaussianKernel5x5[25] =
	{
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
		4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
		2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
	};*/
	const float gaussianKernel5x5[25] = {1.f/273.f,4.f / 273.f,7.f / 273.f,4.f / 273.f,1.f / 273.f,
										4.f / 273.f,16.f / 273.f,26.f / 273.f,16.f / 273.f,4.f / 273.f,
										7.f / 273.f,26.f / 273.f,41.f / 273.f,26.f / 273.f,7.f / 273.f,
										4.f / 273.f,16.f / 273.f,26.f / 273.f,16.f / 273.f,4.f / 273.f, 
										1.f / 273.f,4.f / 273.f,7.f / 273.f,4.f / 273.f,1.f / 273.f};
	cudaMemcpyToSymbol(convolutionGaussianxStore, gaussianKernel5x5, sizeof(gaussianKernel5x5), 0);
	

	// Sobel gradient kernels
	const float sobelGradientX[9] =
	{
		-1.f, 0.f, 1.f,
		-2.f, 0.f, 2.f,
		-1.f, 0.f, 1.f,
	};
	const float sobelGradientY[9] =
	{
		1.f, 2.f, 1.f,
		0.f, 0.f, 0.f,
		-1.f, -2.f, -1.f,
	};
	cudaMemcpyToSymbol(convolutionsobelxStore, sobelGradientX, sizeof(sobelGradientX),0);
	cudaMemcpyToSymbol(convolutionsobelyStore, sobelGradientY, sizeof(sobelGradientY), 0);
	const intptr_t sobelGradientXOffset = sizeof(gaussianKernel5x5) / sizeof(float);
	const intptr_t sobelGradientYOffset = sizeof(sobelGradientX) / sizeof(float) + sobelGradientXOffset;

	// Create CPU/GPU shared images - one for the initial and one for the result
	//camera >> frame;
	//modify
	camera.read(frame);
	//modify
	/*unsigned char *sourceDataDevice, *blurredDataDevice, *edgesDataDevice;
	cv::Mat source(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sourceDataDevice));
	cv::Mat blurred(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &blurredDataDevice));
	cv::Mat edges(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDevice));
	*/
	unsigned char *sourceDataDevice, *sourceDataDevice2, *blurredDataDevice, *edgesDataDevice, *edgesDataDevice_angle, *edgesDataDevice_angle2
		,*edgesDataDeviceThreshold, *edgesDataDeviceThreshold2, *motion_area, *difference, *blurredDataDevice2, *edgesDataDevice2;
	cv::Mat source(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sourceDataDevice));
	cv::Mat source2(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sourceDataDevice2));
	cv::Mat blurred(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &blurredDataDevice));
	cv::Mat blurred2(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &blurredDataDevice2));
	cv::Mat edges(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDevice));
	cv::Mat edges2(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDevice2));
	cv::Mat edges_angle(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDevice_angle));
	cv::Mat edges_angle2(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDevice_angle2));
	cv::Mat edgesThreshold(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDeviceThreshold));
	cv::Mat edgesThreshold2(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDeviceThreshold2));
	cv::Mat motion_area_img(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &motion_area));
	cv::Mat difference_img(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &difference));

	// Create two temporary images (for holding sobel gradients)
	unsigned char *deviceGradientX, *deviceGradientY, *deviceGradientX1, *deviceGradientY1;
	cudaMalloc(&deviceGradientX, frame.size().width * frame.size().height);
	cudaMalloc(&deviceGradientY, frame.size().width * frame.size().height);
	cudaMalloc(&deviceGradientX1, frame.size().width * frame.size().height);
	cudaMalloc(&deviceGradientY1, frame.size().width * frame.size().height);
//modify
cudaStream_t	stream0, stream1;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);
//
	// Loop while capturing images
	while (1)
	{
		// Capture the image and store a gray conversion to the gpu
		//camera >> frame;
		//camera >> frame2;
	//modify
		bool flag = camera.read(frame);
				//Breaking the while loop at the end of the video
				if (flag == false)
				{
					break;
				}
		 flag = camera.read(frame2);
				//Breaking the while loop at the end of the video
				if (flag == false)
				{
					break;
				}

		//cv::Mat bgr[3],finalImg;
		//split(frame, bgr);
		//modify
		//cv::cvtColor(frame, source, cv::COLOR_BGR2GRAY);
		cv::Mat bgr[3], motionFrame;
		split(frame, bgr);
		motionFrame = frame.clone();
		cv::cvtColor(frame, source, cv::COLOR_BGR2GRAY);
		cv::cvtColor(frame2, source2, cv::COLOR_BGR2GRAY);
		if(!seq)
		{ 
			cudaEventRecord(start);
			{
				dim3 cblocks(frame.size().width / frame_size, frame.size().height / frame_size);
				dim3 gausiancblocks(frame.size().width / gaussianframe_size, frame.size().height / gaussianframe_size);
				dim3 cblocks1(frame.size().width / 32, frame.size().height / 32);
				dim3 cblocks2(frame.size().width / 30, frame.size().height / 30);
				dim3 cthreads(32, 32);

				dim3 Blockthreads(block_size, block_size);
				dim3 BlockthreadsGaussian(gaussian_block_size, gaussian_block_size);
				// pythagoran kernel launch paramters
				dim3 pblocks(frame.size().width * frame.size().height / 1024);
				dim3 pthreads(1024, 1);
				//dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
				//dim3 cthreads(16, 16);
				ConvolutionKernel<gaussian_block_size, gaussianframe_size> << <gausiancblocks, BlockthreadsGaussian ,0,stream0>> > (sourceDataDevice, frame.size().height, frame.size().width, 5, blurredDataDevice, 1);
				ConvolutionKernel<block_size, frame_size> << <cblocks, Blockthreads,0,stream0 >> > (blurredDataDevice, frame.size().height, frame.size().width, 3, deviceGradientX, 2);
				ConvolutionKernel<block_size, frame_size> << <cblocks, Blockthreads,0,stream0 >> > (blurredDataDevice, frame.size().height, frame.size().width, 3, deviceGradientY, 3);
				pythagoras<1024> << <pblocks, pthreads,0,stream0 >> > (deviceGradientX, deviceGradientY, edgesDataDevice, edgesDataDevice_angle);
				thresholding_and_suppression<32> << <cblocks2, cthreads,0,stream0 >> > (edgesDataDeviceThreshold, edgesDataDevice, edgesDataDevice_angle, frame.size().width, frame.size().height, 70, 50);
				ConvolutionKernel<gaussian_block_size, gaussianframe_size> << <gausiancblocks, BlockthreadsGaussian ,0,stream1>> > (sourceDataDevice2, frame.size().height, frame.size().width, 5, blurredDataDevice2, 1);
				ConvolutionKernel<block_size, frame_size> << <cblocks, Blockthreads ,0,stream1>> > (blurredDataDevice2, frame.size().height, frame.size().width, 3, deviceGradientX1, 2);
				ConvolutionKernel<block_size, frame_size> << <cblocks, Blockthreads,0,stream1>> > (blurredDataDevice2, frame.size().height, frame.size().width, 3, deviceGradientY1, 3);
				pythagoras<1024> << <pblocks, pthreads,0,stream1>> > (deviceGradientX1, deviceGradientY1, edgesDataDevice2, edgesDataDevice_angle2);
				thresholding_and_suppression<32> << <cblocks2, cthreads,0,stream1>> > (edgesDataDeviceThreshold2, edgesDataDevice2, edgesDataDevice_angle2, frame.size().width, frame.size().height, 70, 50);
				difference_filter<32> << <cblocks1, cthreads >> > (difference, edgesDataDeviceThreshold, edgesDataDeviceThreshold2, frame.size().width, frame.size().height, 0.5);

			}
			cudaEventRecord(stop);

			// Display the elapsed time
			float ms = 0.0f;
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&ms, start, stop);
			totaltime+=ms;
			std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;

		
		}
		else
		{
			const clock_t begin_time = clock();
		//	auto start = std::chrono::steady_clock::now();
			cv::Mat grad_x, grad_y;
			cv::Mat abs_grad_x, abs_grad_y;
		//	cv::size a = 5;
			cv::GaussianBlur(source, blurred, cv::Size(5, 5),0);
			/// Gradient X
			int scale = 1;
			int delta = 0;
 //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
			cv::Sobel(blurred, grad_x, CV_8U, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
			cv::convertScaleAbs(grad_x, abs_grad_x);

			/// Gradient Y
			//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
			cv::Sobel(blurred, grad_y, CV_16S, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
			cv::convertScaleAbs(grad_y, abs_grad_y);

			/// Total Gradient (approximate)
			addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);
			serial_thresholding_and_suppression(edgesThreshold.data, frame.size().width, frame.size().height, abs_grad_x.data, abs_grad_y.data, 70, 50);
//			serial_thresholding_and_suppression(edges);
			// do something
			//
			cv::GaussianBlur(source2, blurred, cv::Size(5, 5), 0);
			/// Gradient X
			//int scale = 1;
			//int delta = 0;
			//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
			cv::Sobel(blurred, grad_x, CV_8U, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
			cv::convertScaleAbs(grad_x, abs_grad_x);

			/// Gradient Y
			//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
			cv::Sobel(blurred, grad_y, CV_16S, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
			cv::convertScaleAbs(grad_y, abs_grad_y);

			/// Total Gradient (approximate)
			addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);
			serial_thresholding_and_suppression(edgesThreshold2.data, frame.size().width, frame.size().height, abs_grad_x.data, abs_grad_y.data, 70, 50);
			//		
			serial_difference_filter(difference, edgesThreshold.data, edgesThreshold2.data, frame.size().width, frame.size().height,0.5);
			//auto end = std::chrono::steady_clock::now();

			// Store the time difference between start and end
			//std::chrono::duration<double> diff = end - start;
			//If you want to print the time difference between start and end in the above code, you could use :
			totaltime+=float(clock() - begin_time) / CLOCKS_PER_SEC;
			std::cout<<"total time taken" << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;
			//std::cout << std::chrono::duration <double>(diff).count() << " s" << std::endl;
		}
		// Record the time it takes to process
		
		// Show the results
		//modify the code
		
	//	cv::Mat edges1 = edges.clone();
//		cv::threshold(edges1, edges1, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		//cv::threshold(edges1, edges1, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		int erosion_size = 3;
		cv::Mat elementErode = getStructuringElement(cv::MORPH_CROSS,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		//erosion_size = 2;
		cv::erode(difference_img, difference_img, elementErode);
		

			int max = 0, min = 255;

			for (int i = 0; i < difference_img.rows; i++) {
				for (int j = 0; j < difference_img.cols; j++) {
					if (difference_img.at<uchar>(i, j) == 255) //{//&&cnt<40) 
					{
						cv::rectangle(motionFrame, cv::Point(j - 5, i - 5), cv::Point(j + 5, i + 5), cv::Scalar(255, 0, 0), 1);
					}
				}
			}

	//		cv::imshow("Source", frame);
		//	cv::imshow("Greyscale", source);
			//cv::imshow("Blurred", blurred);
			//cv::imshow("Sobel", edges);
			//cv::imshow("Threshold", edgesThreshold);
			//cv::imshow("Difference", difference_img);
			//cv::imshow("Motion", motionFrame);
		if (cv::waitKey(1) == 27) break;
	}

	// Exit
	cudaFreeHost(source.data);
	cudaFreeHost(blurred.data);
	cudaFreeHost(edges.data);
	cudaFree(deviceGradientX);
	cudaFree(deviceGradientY);
std::cout<<"totaltimetaken"<<totaltime;
	return 0;
}
