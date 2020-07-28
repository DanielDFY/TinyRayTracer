#include <iostream>
#include <fstream>
#include <ctime>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

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

#include <helperUtils.cuh>
#include <curand_kernel.h>

using namespace TinyRT;

constexpr int objNum = 1;
constexpr int imageTexNum = 1;

struct TextureData {
	cudaTextureObject_t textureObject;
	int width, height;
	cudaArray* deviceData;
};

TextureData loadAndInitTexture(const char* fileName) {
	int width, height, depth;
	const auto texData = stbi_load(fileName, &width, &height, &depth, 0);
	const auto pixelNum = width * height;
	const auto imageSize = pixelNum * depth;
	float* hostData = new float[imageSize];
	for (unsigned int layer = 0; layer < 3; layer++)
		for (auto i = 0; i < pixelNum; i++)
			hostData[layer * pixelNum + i] = texData[i * 3 + layer] / 255.0f;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray* deviceData;
	cudaMalloc3DArray(&deviceData, &channelDesc, make_cudaExtent(width, height, 3), cudaArrayLayered);

	cudaMemcpy3DParms memcpy3DParms = { 0 };
	memcpy3DParms.srcPos = make_cudaPos(0, 0, 0);
	memcpy3DParms.dstPos = make_cudaPos(0, 0, 0);
	memcpy3DParms.srcPtr = make_cudaPitchedPtr(hostData, width * sizeof(float), width, height);
	memcpy3DParms.dstArray = deviceData;
	memcpy3DParms.extent = make_cudaExtent(width, height, 3);
	memcpy3DParms.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&memcpy3DParms);

	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = deviceData;
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = true;

	cudaTextureObject_t textureObject;
	cudaCreateTextureObject(&textureObject, &texRes, &texDesc, nullptr);

	return { textureObject, width, height, deviceData };
}

void cleanTexture(TextureData textureData) {
	cudaDestroyTextureObject(textureData.textureObject);
	cudaFreeArray(textureData.deviceData);
}

__device__ void earth(Hittable** hittablePtrList, curandState* const objRandStatePtr, Texture** texturePtrList, TextureData* textureDataList) {
	size_t objIdx = 0;
	size_t textureIdx = 0;

	TextureData earthTextureData = textureDataList[textureIdx];
	texturePtrList[textureIdx] = new ImageTexture(earthTextureData.textureObject, earthTextureData.width, earthTextureData.height);
	hittablePtrList[objIdx] = new Sphere(Point3(0, 0, 0), 2, new Lambertian(texturePtrList[textureIdx]));
}

__device__ Color rayColor(const Ray& r, Hittable** hittablePtr, const int maxDepth, curandState* const randStatePtr) {
	Ray curRay = r;
	Vec3 curAttenuation(1.0f, 1.0f, 1.0f);
	for (size_t i = 0; i < maxDepth; ++i) {
		HitRecord rec;
		if ((*hittablePtr)->hit(curRay, 0.001f, M_FLOAT_INFINITY, rec)) {
			Ray scattered;
			Vec3 attenuation;
			if (rec.matPtr->scatter(curRay, rec, attenuation, scattered, randStatePtr)) {
				curRay = scattered;
				curAttenuation *= attenuation;
			} else {
				return { 0.0f, 0.0f, 0.0f };
			}
		} else {
			const Vec3 unitDirection = unitVec3(curRay.direction());
			const double t = 0.5f * (unitDirection.y() + 1.0f);
			const Color background = (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
			return curAttenuation * background;
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

		pixelColor += rayColor(r, hittablePtrList, maxDepth, &randState);
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

__global__ void createWorld(Camera** camera, float aspectRatio, Hittable** hittablePtrList, Hittable** hittableWorldObjListPtr, curandState* objRandStatePtr, Texture** texturePtrList, TextureData* textureDataList) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		const Point3 lookFrom(13.0f, 2.0f, -3.0f);
		const Point3 lookAt(0.0f, 0.0f, 0.0f);
		const Vec3 vUp(0.0f, 1.0f, 0.0f);
		const float vFov = 20.0f;
		const float aperture = 0.0f;
		const float distToFocus = 10.0f;
		const float time0 = 0.0f;
		const float time1 = 1.0f;
		
		*camera = new Camera(lookFrom, lookAt, vUp, vFov, aspectRatio, aperture, distToFocus, time0, time1);

		earth(hittablePtrList, objRandStatePtr, texturePtrList, textureDataList);

		*hittableWorldObjListPtr = new HittableList(hittablePtrList, objNum);
	}
}

__global__ void freeWorld(Camera** camera, Hittable** hittableList, size_t hittableNum, Hittable** hittableWorldObjList, Texture** texturePtrList) {
	delete* camera;
	for (int i = 0; i < hittableNum; ++i) {
		// delete material instances
		delete hittableList[i]->matPtr();
		// delete object instances
		delete hittableList[i];
	}

	for (int i = 0; i < imageTexNum; ++i) {
		// delete texture instances
		delete texturePtrList[i];
	}
	
	delete* hittableWorldObjList;
}

int main() {
	/* image config */
	constexpr float aspectRatio = 16.0f / 9.0f;
	constexpr int imageWidth = 800;
	constexpr int imageHeight = static_cast<int>(imageWidth / aspectRatio);
	constexpr int samplesPerPixel = 20;
	constexpr int maxDepth = 5;

	/* image output file */
	const std::string fileName("output.png");

	/* thread block config */
	constexpr int threadBlockWidth = 16;
	constexpr int threadBlockHeight = 16;

	// preparation
	constexpr int channelNum = 3; // rgb
	constexpr int pixelNum = imageWidth * imageHeight;

	// allocate memory for pixel buffer
	const auto pixelBufferPtr = cudaManagedUniquePtr<Color>(pixelNum * sizeof(Color));

	// allocate random state
	const auto seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
	const auto objRandStatePtr = cudaUniquePtr<curandState>(sizeof(curandState));
	const auto pixelRandStateListPtr = cudaUniquePtr<curandState>(pixelNum * sizeof(curandState));

	// create world of hittable objects and the camera
	const auto cameraPtr = cudaUniquePtr<Camera*>(sizeof(Camera*));
	const auto hittablePtrList = cudaUniquePtr<Hittable*>(objNum * sizeof(Hittable*));
	const auto hittableWorldObjListPtr = cudaUniquePtr<Hittable*>(sizeof(Hittable*));

	const auto texturePtrList = cudaUniquePtr<Texture*>(imageTexNum * sizeof(Texture*));
	const auto textureDataList = cudaManagedUniquePtr<TextureData>(imageTexNum * sizeof(TextureData));

	// load and initiate earth texture
	textureDataList.get()[0] = loadAndInitTexture("earthmap.jpg");
	
	createInit<<<1, 1>>>(objRandStatePtr.get(), seed);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	createWorld<<<1, 1>>>(cameraPtr.get(), aspectRatio, hittablePtrList.get(), hittableWorldObjListPtr.get(), objRandStatePtr.get(), texturePtrList.get(), textureDataList.get());
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

	// free resources
	freeWorld<<<1, 1>>>(cameraPtr.get(), hittablePtrList.get(), objNum, hittableWorldObjListPtr.get(), texturePtrList.get());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	//cleanTexture(earthTextureData);
	
	return 0;
}