#include <iostream>
#include <fstream>
#include <ctime>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Color.cuh>
#include <Ray.cuh>
#include <Camera.cuh>
#include <Sphere.cuh>
#include <HittableList.cuh>
#include <Material.cuh>

#include <helperUtils.h>
#include <curand_kernel.h>

using namespace TinyRT;

__device__ Color rayColor(const Ray& r, Hittable** hittable, const int maxDepth, curandState* const randStatePtr) {
	Ray curRay = r;
	Vec3 curAttenuation(1.0f, 1.0f, 1.0f);
	for (size_t i = 0; i < maxDepth; ++i) {
		HitRecord rec;
		if ((*hittable)->hit(curRay, 0.0001f, M_FLOAT_INFINITY, rec)) {
			Ray scattered;
			Vec3 attenuation;
			if (rec.matPtr->scatter(curRay, rec, attenuation, scattered, randStatePtr)) {
				curRay = scattered;
				curAttenuation *= attenuation;
			}
			else {
				return { 0.0, 0.0, 0.0 };
			}
		}
		else {
			const Vec3 unitDirection = unitVec3(curRay.direction());
			const double t = 0.5f * (unitDirection.y() + 1.0f);
			const Color background = (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
			return curAttenuation * background;
		}
	}
	// exceed max depth
	return { 0.0f, 0.0f, 0.0f };
}

__global__ void renderInit(const int imageWidth, const int imageHeight, curandState* const randStateList) {
	const int col = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = threadIdx.y + blockIdx.y * blockDim.y;
	if ((col >= imageWidth) || (row >= imageHeight))
		return;

	const int idx = row * imageWidth + col;

	// init random numbers for anti-aliasing
	// each thread gets its own special seed, fixed sequence number, fixed offset
	curand_init(2020 + idx, 0, 0, &randStateList[idx]);
}

__global__ void render(
	Color* const pixelBuffer,
	const int imageWidth,
	const int imageHeight,
	Camera** const camera,
	curandState* const randStateList,
	const int samplesPerPixel,
	const int maxDepth,
	Hittable** const hittableWorldObjList) {

	const int col = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col >= imageWidth || row >= imageHeight)
		return;

	const int idx = row * imageWidth + col;

	curandState randState = randStateList[idx];
	Color pixelColor(0.0f, 0.0f, 0.0f);
	for (size_t s = 0; s < samplesPerPixel; ++s) {
		const auto u = (static_cast<float>(col) + curand_uniform(&randState)) / static_cast<float>(imageWidth - 1);
		const auto v = 1.0 - (static_cast<float>(row) + curand_uniform(&randState)) / static_cast<float>(imageHeight - 1);

		const Ray r = (*camera)->getRay(u, v);

		pixelColor += rayColor(r, hittableWorldObjList, maxDepth, &randState);
	}

	pixelColor /= samplesPerPixel;
	pixelColor.gammaCorrect();

	pixelBuffer[idx] = pixelColor;
}

__global__ void createWorld(Camera** camera, float aspectRatio, Hittable** hittableList, Hittable** hittableWorldObjList) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*camera = new Camera(Point3(-2, 2, -1), Point3(0, 0, 1), Vec3(0, 1, 0), 20, aspectRatio);
		hittableList[0] = new Sphere(Point3(0.0f, 0.0f, 1.0f), 0.5f, new Lambertian(Color(0.1f, 0.2f, 0.5f)));
		hittableList[1] = new Sphere(Point3(0.0f, -100.5f, 1.0f), 100.0f, new Lambertian(Color(0.8f, 0.8f, 0.0f)));
		hittableList[2] = new Sphere(Point3(1.0f, 0.0f, 1.0f), 0.5f, new Metal(Color(0.8f, 0.8f, 0.8f), 0.5f));
		hittableList[3] = new Sphere(Point3(-1.0f, 0.0f, 1.0f), 0.5f, new Dielectric(1.5f));
		hittableList[4] = new Sphere(Point3(-1.0f, 0.0f, 1.0f), -0.45f, new Dielectric(1.5f));
		*hittableWorldObjList = new HittableList(hittableList, 5);
	}
}

__global__ void freeWorld(Camera** camera, Hittable** hittableList, Hittable** hittableWorldObjList) {
	delete* camera;
	for (int i = 0; i < 5; ++i) {
		delete (static_cast<Sphere*>(hittableList[i]))->matPtr();
		delete hittableList[i];
	}
	delete* hittableWorldObjList;
}

int main() {
	/* image config */
	constexpr float aspectRatio = 16.0f / 9.0f;
	constexpr int imageWidth = 400;
	constexpr int imageHeight = static_cast<int>(imageWidth / aspectRatio);
	constexpr int samplesPerPixel = 100;
	constexpr int maxDepth = 50;

	/* image output file */
	const std::string fileName("output.png");

	/* thread block config */
	constexpr int threadBlockWidth = 8;
	constexpr int threadBlockHeight = 8;

	// preparation
	constexpr int channelNum = 3; // rgb
	constexpr int pixelNum = imageWidth * imageHeight;
	constexpr size_t pixelBufferBytes = pixelNum * sizeof(Color);
	constexpr size_t randStateListBytes = pixelNum * sizeof(curandState);

	// allocate memory for pixel buffer
	const auto pixelBufferPtr = cudaManagedUniquePtr<Color>(pixelBufferBytes);

	// allocate random state
	const auto randStateListPtr = cudaUniquePtr<curandState>(randStateListBytes);

	// create world of hittable objects and the camera
	const auto cameraPtr = cudaUniquePtr<Camera*>(sizeof(Camera*));
	const auto hittableListPtr = cudaUniquePtr<Hittable*>(5 * sizeof(Hittable*));
	const auto hittableWorldObjListPtr = cudaUniquePtr<Hittable*>(sizeof(Hittable*));
	createWorld<<<1, 1>>>(cameraPtr.get(), aspectRatio, hittableListPtr.get(), hittableWorldObjListPtr.get());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// start timer
	const clock_t start = clock();

	const dim3 blockDim(imageWidth / threadBlockWidth + 1, imageHeight / threadBlockHeight + 1);
	const dim3 threadDim(threadBlockWidth, threadBlockHeight);

	// render init
	renderInit<<<blockDim, threadDim>>>(imageWidth, imageHeight, randStateListPtr.get());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// render the image into buffer
	render<<<blockDim, threadDim>>>(
		pixelBufferPtr.get(),
		imageWidth,
		imageHeight,
		cameraPtr.get(),
		randStateListPtr.get(),
		samplesPerPixel,
		maxDepth,
		hittableWorldObjListPtr.get()
	);
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
	freeWorld<<<1, 1>>>(cameraPtr.get(), hittableListPtr.get(), hittableWorldObjListPtr.get());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	return 0;
}