#include <iostream>
#include <fstream>
#include <ctime>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Color.cuh>
#include <Ray.cuh>
#include <Sphere.cuh>
#include <HittableList.cuh>

#include <helperCuda.h>

using namespace TinyRT;

__device__ Color rayColor(const Ray& r, Hittable** hittable) {
	HitRecord rec;
	if ((*hittable)->hit(r, 0.0f, DOUBLE_INFINITY, rec)) {
		return 0.5 * Color(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, -rec.normal.z() + 1.0f);
	}

	const Vec3 unitDirection = unitVec3(r.direction());
	const float t = 0.5f * (unitDirection.y() + 1.0f);
	return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
}

__global__ void render(
	Color* pixelBuffer,
	int imageWidth, int
	imageHeight,
	Vec3 lowerLeftCorner,
	Vec3 horizontal,
	Vec3 vertical,
	Vec3 origin,
	Hittable** hittableWorldObjs) {
	const int col = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col >= imageWidth || row >= imageHeight)
		return;

	const int idx = row * imageWidth + col;

	const auto u = static_cast<float>(col) / static_cast<float>(imageWidth - 1);
	const auto v = 1.0f - static_cast<float>(row) / static_cast<float>(imageHeight - 1);

	const Ray r(origin, lowerLeftCorner - origin + u * horizontal + v * vertical);

	pixelBuffer[idx] = rayColor(r, hittableWorldObjs);
}

__global__ void createWorld(Hittable** hittableList, Hittable** hittableWorldObjs) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*hittableList = new Sphere(Vec3(0.0f, 0.0f, 1.0f), 0.5f);
		*(hittableList + 1) = new Sphere(Vec3(0.0f, -100.5f, 1.0f), 100.0f);
		*hittableWorldObjs = new HittableList(hittableList, 2);
	}
}

__global__ void freeWorld(Hittable** hittableList, Hittable** hittableWorldObjs) {
	delete *hittableList;
	delete *(hittableList + 1);
	delete *hittableWorldObjs;
}

int main() {
	/* image config */
	constexpr float aspectRatio = 16.0f / 9.0f;
	constexpr int imageWidth = 400;
	constexpr int imageHeight = static_cast<int>(imageWidth / aspectRatio);

	/* image output file */
	const std::string fileName("output.png");

	/* thread block config */
	constexpr int threadBlockWidth = 8;
	constexpr int threadBlockHeight = 8;

	// preparation
	constexpr int channelNum = 3; // rgb
	constexpr int pixelNum = imageWidth * imageHeight;
	constexpr size_t pixelBufferBytes = pixelNum * sizeof(Color);

	// allocate memory for pixel buffer
	const auto pixelBufferPtr = cudaManagedUniquePtr<Color>(pixelBufferBytes);

	/* camera config */
	constexpr float viewPortHeight = 2.0f;
	constexpr float viewPortWidth = aspectRatio * viewPortHeight;
	constexpr float focalLength = 1.0f;

	const Point3 origin(0.0f, 0.0f, 0.0f);
	const Vec3 horizontal(viewPortWidth, 0.0f, 0.0f);
	const Vec3 vertical(0.0f, viewPortHeight, 0.0f);
	// left-handed Y up
	const Point3 lowerLeftCorner = origin - horizontal / 2 - vertical / 2 + Vec3(0.0f, 0.0f, focalLength);

	// create world of hittable objects
	const auto hittableListPtr = cudaUniquePtr<Hittable*>(2 * sizeof(Hittable*));
	const auto hittableWorldObjsPtr = cudaUniquePtr<Hittable*>(sizeof(Hittable*));
	createWorld<<<1, 1>>>(hittableListPtr.get(), hittableWorldObjsPtr.get());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	// start timer
	const clock_t start = clock();

	const dim3 blockDim(imageWidth / threadBlockWidth + 1, imageHeight / threadBlockHeight + 1);
	const dim3 threadDim(threadBlockWidth, threadBlockHeight);

	// render the image into buffer
	render<<<blockDim, threadDim>>>(pixelBufferPtr.get(), imageWidth, imageHeight, lowerLeftCorner, horizontal, vertical, origin, hittableWorldObjsPtr.get());
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
		const Color color = pixelBufferPtr.get()[pixelIdx];
		pixelDataPtr[dataIdx++] = static_cast<unsigned char>(color.r8bit());
		pixelDataPtr[dataIdx++] = static_cast<unsigned char>(color.g8bit());
		pixelDataPtr[dataIdx++] = static_cast<unsigned char>(color.b8bit());
	}

	// print rendering time
	std::cout << "Complete!\n" << "The rendering took " << renderingMillisecond << "ms" << std::endl;

	// write pixel data to output file
	stbi_write_png(fileName.c_str(), imageWidth, imageHeight, channelNum, pixelDataPtr.get(), strideBytes);

	// free world of hittable objects
	freeWorld<<<1, 1>>>(hittableListPtr.get(), hittableWorldObjsPtr.get());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	return 0;
}