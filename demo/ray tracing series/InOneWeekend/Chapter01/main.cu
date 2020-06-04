#include <fstream>
#include <ctime>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <helper_cuda.h>

// image channel number
#define CHANNEL_NUM 3	// rgb

__global__
void render(float* imageBuffer, int imageWidth, int imageHeight) {
	const int col = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col >= imageWidth || row >= imageHeight)
		return;

	const int idx = (row * imageWidth + col) * CHANNEL_NUM;

	imageBuffer[idx] = static_cast<float>(row) / static_cast<float>(imageHeight - 1);
	imageBuffer[idx + 1] = static_cast<float>(col) / static_cast<float>(imageWidth - 1);
	imageBuffer[idx + 2] = 0.25f;
}

int main() {
	/* image config */
	constexpr int imageWidth = 400;
	constexpr int imageHeight = 300;

	/* image output file */
	const std::string fileName("output.png");

	/* thread block config */
	constexpr int threadBlockWidth = 8;
	constexpr int threadBlockHeight = 8;

	// preparation
	constexpr int pixelNum = imageWidth * imageHeight;
	const size_t imageBufferSize = pixelNum * CHANNEL_NUM * sizeof(float);

	float* imageBuffer;
	checkCudaErrors(cudaMallocManaged(&imageBuffer, imageBufferSize));

	dim3 blockDim(imageWidth / threadBlockWidth + 1, imageHeight / threadBlockHeight + 1);
	dim3 threadDim(threadBlockWidth, threadBlockHeight);

	// measure rendering time
	clock_t start, stop;
	start = clock();
	
	// render the image into buffer
	render<<<blockDim, threadDim>>>(imageBuffer, imageWidth, imageHeight);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	stop = clock();
	const auto renderingMillisecond = stop - start;

	// other image writer arguments
	constexpr int imageSize = pixelNum * CHANNEL_NUM;
	constexpr size_t strideInBytes = imageWidth * CHANNEL_NUM * sizeof(unsigned char);
	const std::unique_ptr<unsigned char[]> pixelDataPtr(new unsigned char[imageSize]);

	// store the pixel data into writing buffer as 8bit color
	for (int idx = 0; idx < imageSize; ++idx) {
		const auto value = imageBuffer[idx];
		pixelDataPtr[idx] = static_cast<unsigned char>(255.999 * value);
	}

	// print rendering time 
	std::cout << "Complete!\n" << "The rendering took " << renderingMillisecond << "ms" << std::endl;

	// write pixel data to output file
	stbi_write_png(fileName.c_str(), imageWidth, imageHeight, CHANNEL_NUM, pixelDataPtr.get(), strideInBytes);

	// clean
	checkCudaErrors(cudaFree(imageBuffer));

	return 0;
}