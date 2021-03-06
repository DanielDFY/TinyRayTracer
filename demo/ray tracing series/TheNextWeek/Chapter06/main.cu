#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Color.cuh>
#include <Ray.cuh>
#include <Camera.cuh>
#include <HittableList.cuh>
#include <Material.cuh>
#include <AARect.cuh>

#include <helperUtils.cuh>
#include <curand_kernel.h>

using namespace TinyRT;

constexpr int objNum = 6;

__device__ void cornellBox(Hittable** hittablePtrList, Texture** extraTexturePtrList) {
	size_t objIdx = 0;
	size_t textureIdx = 0;

	Texture* redTex = new SolidColor(0.65f, 0.05f, 0.05f);
	extraTexturePtrList[textureIdx++] = redTex;
	Texture* whiteTex = new SolidColor(0.73f, 0.73f, 0.73f);
	extraTexturePtrList[textureIdx++] = whiteTex;
	Texture* greenTex = new SolidColor(0.12f, 0.45f, 0.15f);
	extraTexturePtrList[textureIdx++] = greenTex;
	Texture* lightTex = new SolidColor(15.0f, 15.0f, 15.0f);
	extraTexturePtrList[textureIdx] = lightTex;

	hittablePtrList[objIdx++] = new YZRect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, new Lambertian(greenTex));
	hittablePtrList[objIdx++] = new YZRect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, new Lambertian(redTex));
	hittablePtrList[objIdx++] = new XZRect(213.0f, 343.0f, 227.0f, 332.0f, 554.0f, new DiffuseLight(lightTex));
	hittablePtrList[objIdx++] = new XZRect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, new Lambertian(whiteTex));
	hittablePtrList[objIdx++] = new XZRect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, new Lambertian(whiteTex));
	hittablePtrList[objIdx] = new XYRect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, new Lambertian(whiteTex));
}

__device__ Color rayColor(const Ray& r, const Color& background, Hittable** hittablePtr, const int maxDepth, curandState* const randStatePtr) {
	Ray curRay = r;
	Vec3 curAttenuation(1.0f, 1.0f, 1.0f);
	Color curEmitted(0.0f, 0.0f, 0.0f);

	for (size_t i = 0; i < maxDepth; ++i) {
		HitRecord rec;
		if ((*hittablePtr)->hit(curRay, 0.001f, M_FLOAT_INFINITY, rec)) {
			Ray scattered;
			Vec3 attenuation;
			Color emitted = rec.matPtr->emitted(rec.u, rec.v, rec.point);

			if (rec.matPtr->scatter(curRay, rec, attenuation, scattered, randStatePtr)) {
				curRay = scattered;
				curEmitted += curAttenuation * emitted;
				curAttenuation *= attenuation;
			} else {
				return curEmitted + curAttenuation * emitted;
			}
		} else {
			return curEmitted + curAttenuation * background;
		}
	}
	// exceed max depth
	return { 0.0f, 0.0f, 0.0f };
}

__global__ void renderInit(const int imageWidth, const int imageHeight, curandState* const randStateList, unsigned int seed) {
	const int col = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = threadIdx.y + blockIdx.y * blockDim.y;
	if ((col >= imageWidth) || (row >= imageHeight))
		return;

	const int idx = row * imageWidth + col;

	// init random numbers for anti-aliasing
	// each thread gets its own special seed, fixed sequence number, fixed offset
	curand_init(seed + idx, 0, 0, &randStateList[idx]);
}

__global__ void render(
	Color* const pixelBuffer,
	const int imageWidth,
	const int imageHeight,
	Camera** const camera,
	const Color background,
	curandState* const pixelRandStateList,
	const int samplesPerPixel,
	const int maxDepth,
	Hittable** const hittablePtrList) {

	const int col = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col >= imageWidth || row >= imageHeight)
		return;

	const int idx = row * imageWidth + col;

	curandState randState = pixelRandStateList[idx];
	Color pixelColor(0.0f, 0.0f, 0.0f);
	for (size_t s = 0; s < samplesPerPixel; ++s) {
		const auto u = (static_cast<float>(col) + randomFloat(&randState)) / static_cast<float>(imageWidth - 1);
		const auto v = 1.0 - (static_cast<float>(row) + randomFloat(&randState)) / static_cast<float>(imageHeight - 1);

		const Ray r = (*camera)->getRay(u, v, &randState);

		pixelColor += rayColor(r, background, hittablePtrList, maxDepth, &randState);
	}

	pixelColor /= samplesPerPixel;
	pixelColor.gammaCorrect();

	pixelBuffer[idx] = pixelColor;
}

__global__ void createInit(curandState* const randStatePtr, unsigned int seed) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		// init a random number for sphere generating
		curand_init(seed, 0, 0, randStatePtr);
	}
}

__global__ void createWorld(Camera** camera, float aspectRatio, Hittable** hittablePtrList, Hittable** hittableWorldObjListPtr, Texture** extraTexturePtrList) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		const Point3 lookFrom(278.0, 278.0, -800.0f);
		const Point3 lookAt(278.0f, 278.0f, 0.0f);
		const Vec3 vUp(0.0f, 1.0f, 0.0f);
		const float vFov = 40.0f;
		const float aperture = 0.0f;
		const float distToFocus = 10.0f;
		const float time0 = 0.0f;
		const float time1 = 0.0f;

		*camera = new Camera(lookFrom, lookAt, vUp, vFov, aspectRatio, aperture, distToFocus, time0, time1);

		cornellBox(hittablePtrList, extraTexturePtrList);

		*hittableWorldObjListPtr = new HittableList(hittablePtrList, objNum);
	}
}

__global__ void freeWorld(Camera** camera, Hittable** hittableList, size_t hittableNum, Hittable** hittableWorldObjList, Texture** extraTexturePtrList, size_t extraTexturePtrNum) {
	delete* camera;
	for (int i = 0; i < hittableNum; ++i) {
		// delete random material instances
		delete hittableList[i]->matPtr();
		// delete object instances
		delete hittableList[i];
	}

	for (int i = 0; i < extraTexturePtrNum; ++i) {
		// delete extra texture instances
		delete extraTexturePtrList[i];
	}

	delete* hittableWorldObjList;
}

int main() {
	/* image config */
	constexpr float aspectRatio = 1.0f;
	constexpr int imageWidth = 800;
	constexpr int imageHeight = static_cast<int>(imageWidth / aspectRatio);
	constexpr int samplesPerPixel = 200;
	constexpr int maxDepth = 50;

	const Color background(0.0f, 0.0f, 0.0f);

	/* image output file */
	const std::string fileName("output.png");

	/* thread block config */
	constexpr int threadBlockWidth = 16;
	constexpr int threadBlockHeight = 16;

	// preparation
	constexpr int channelNum = 3; // rgb
	constexpr int pixelNum = imageWidth * imageHeight;
	constexpr size_t pixelBufferBytes = pixelNum * sizeof(Color);
	constexpr size_t randStateListBytes = pixelNum * sizeof(curandState);

	// allocate memory for pixel buffer
	const auto pixelBufferPtr = cudaManagedUniquePtr<Color>(pixelBufferBytes);

	// allocate random state
	const auto seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
	const auto objRandStatePtr = cudaUniquePtr<curandState>(sizeof(curandState));
	const auto pixelRandStateListPtr = cudaUniquePtr<curandState>(randStateListBytes);

	// create world of hittable objects and the camera
	const auto cameraPtr = cudaUniquePtr<Camera*>(sizeof(Camera*));
	const auto hittablePtrList = cudaUniquePtr<Hittable*>(objNum * sizeof(Hittable*));
	const auto hittableWorldObjListPtr = cudaUniquePtr<Hittable*>(sizeof(Hittable*));

	constexpr size_t extraTexturePtrNum = 1;
	const auto extraTexturePtrList = cudaUniquePtr<Texture*>(extraTexturePtrNum * sizeof(Texture*));

	createInit<<<1, 1>>>(objRandStatePtr.get(), seed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	createWorld<<<1, 1>>>(cameraPtr.get(), aspectRatio, hittablePtrList.get(), hittableWorldObjListPtr.get(), extraTexturePtrList.get());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// start timer
	const clock_t start = clock();

	const dim3 blockDim(imageWidth / threadBlockWidth + 1, imageHeight / threadBlockHeight + 1);
	const dim3 threadDim(threadBlockWidth, threadBlockHeight);

	// render init
	renderInit<<<blockDim, threadDim>>>(imageWidth, imageHeight, pixelRandStateListPtr.get(), seed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// render the image into buffer
	render<<<blockDim, threadDim>>>(
		pixelBufferPtr.get(),
		imageWidth,
		imageHeight,
		cameraPtr.get(),
		background,
		pixelRandStateListPtr.get(),
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
	freeWorld<<<1, 1>>>(cameraPtr.get(), hittablePtrList.get(), objNum, hittableWorldObjListPtr.get(), extraTexturePtrList.get(), extraTexturePtrNum);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	return 0;
}