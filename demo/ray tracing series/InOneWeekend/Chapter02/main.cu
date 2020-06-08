#include <iostream>
#include <fstream>
#include <ctime>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Color.cuh>

#include <helperUtils.h>

using namespace TinyRT;

__global__ void render(Color* const pixelBuffer, const int imageWidth, const int imageHeight) {
	const int col = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col >= imageWidth || row >= imageHeight)
		return;

	const int idx = row * imageWidth + col;

	pixelBuffer[idx] = Color(
		static_cast<float>(row) / static_cast<float>(imageHeight - 1),
		static_cast<float>(col) / static_cast<float>(imageWidth - 1),
		0.25f);
}

int main() {
	/* image config */
	constexpr int imageWidth = 400;
	constexpr int imageHeight = 250;

	/* image output file */
	const std::string fileName("output.png");

	/* thread block config */
	constexpr int threadBlockWidth = 8;
	constexpr int threadBlockHeight = 8;

	// preparation
	constexpr int channelNum = 3; // rgb
	constexpr int pixelNum = imageWidth * imageHeight;
	constexpr size_t pixelBufferBytes = pixelNum * sizeof(Color);

	Color* pixelBuffer;
	checkCudaErrors(cudaMallocManaged(&pixelBuffer, pixelBufferBytes));

	// start timer
	const clock_t start = clock();

	dim3 blockDim(imageWidth / threadBlockWidth + 1, imageHeight / threadBlockHeight + 1);
	dim3 threadDim(threadBlockWidth, threadBlockHeight);

	// render the image into buffer
	render<<<blockDim, threadDim>>>(pixelBuffer, imageWidth, imageHeight);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// stop timer
	const clock_t stop = clock();

	// measure rendering time
	const auto renderingMillisecond = stop - start;

	// other image writer arguments
	constexpr int imageSize = pixelNum * channelNum;
	constexpr size_t strideBytes = imageWidth * channelNum * sizeof(unsigned char);
	const std::unique_ptr<unsigned char[]> pixelDataPtr(new unsigned char[imageSize]);

	// store the pixel data into writing buffer as 8bit color
	for (int pixelIdx = 0, dataIdx = 0; pixelIdx < pixelNum; ++pixelIdx) {
		const Color color = pixelBuffer[pixelIdx];
		pixelDataPtr[dataIdx++] = static_cast<unsigned char>(color.r8bit());
		pixelDataPtr[dataIdx++] = static_cast<unsigned char>(color.g8bit());
		pixelDataPtr[dataIdx++] = static_cast<unsigned char>(color.b8bit());
	}

	// print rendering time
	std::cout << "Complete!\n" << "The rendering took " << renderingMillisecond << "ms" << std::endl;

	// write pixel data to output file
	stbi_write_png(fileName.c_str(), imageWidth, imageHeight, channelNum, pixelDataPtr.get(), strideBytes);

	// clean
	checkCudaErrors(cudaFree(pixelBuffer));

	return 0;
}