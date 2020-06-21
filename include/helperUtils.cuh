#pragma once

#include <iostream>
#include <memory>
#include <chrono>
#include <limits>
#include <random>
#include <cuda_runtime_api.h>

namespace TinyRT {
	/* Math Utils */
	constexpr float M_FLOAT_INFINITY = std::numeric_limits<float>::infinity();
	constexpr float M_PI = 3.1415926535897932385f;

	__host__ __device__ inline float degreeToRadian(const float degree) { return degree * M_PI / 180.0f; }

	inline float randomFloat() {
		// returns a random real in [0,1)
		const auto seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
		static std::mt19937 generator(seed);
		static std::uniform_real_distribution<float> distribution(0.0, 1.0);
		return distribution(generator);
	}

	inline float randomFloat(float min, float max) {
		// returns a random real in [min,max)
		return min + (max - min) * randomFloat();
	}

	inline int randomInt(int min, int max) {
		// Returns a random integer in [min,max].
		return static_cast<int>(randomFloat(static_cast<float>(min), static_cast<float>(max) + 1.0f));
	}

	__device__ inline float randomFloat(curandState* randStatePtr) {
		return curand_uniform(randStatePtr);
	}

	__device__ inline float randomFloat(float min, float max, curandState* randStatePtr) {
		return min + (max - min) * randomFloat(randStatePtr);
	}

	__device__ inline int randomInt(int min, int max, curandState* randStatePtr) {
		// Returns a random integer in [min,max].
		return static_cast<int>(randomFloat(min, max + 1.0f, randStatePtr));
	}

	inline float clamp(float x, float min, float max) {
		if (x < min) return min;
		if (x > max) return max;
		return x;
	}
	
	/* Error Checking */
	#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

	template<typename T>
	void check(const T err, const char* func, const char* file, int line) {
		if (err != cudaSuccess) {
			std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
			std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
			// Make sure we call CUDA Device Reset before exiting
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	}

	/* Pointer Wrappers */
	
	// allocator wrappers for using smart pointer
	const auto cudaMallocWrapper = [](const size_t sizeInBytes) {
		void* ptr;
		checkCudaErrors(cudaMalloc(static_cast<void**>(&ptr), sizeInBytes));
		return ptr;
	};
	
	const auto cudaMallocManagedWrapper = [](const size_t sizeInBytes) {
		void* ptr;
		checkCudaErrors(cudaMallocManaged(static_cast<void**>(&ptr), sizeInBytes));
		return ptr;
	};

	// deleter wrapper for using smart pointer
	template<typename T>
	const auto cudaDeleterWrapper = [](T* ptr) {
		checkCudaErrors(cudaFree(ptr));
	};

	// wrapped smart pointers
	template<typename T>
	auto cudaUniquePtr = [](const size_t sizeInBytes) {
		return std::unique_ptr<T, decltype(cudaDeleterWrapper<T>)>(
			static_cast<T*>(cudaMallocWrapper(sizeInBytes)),
			cudaDeleterWrapper<T>);
	};;
	
	template<typename T>
	auto cudaManagedUniquePtr = [](const size_t sizeInBytes) {
		return std::unique_ptr<T, decltype(cudaDeleterWrapper<T>)>(
			static_cast<T*>(cudaMallocManagedWrapper(sizeInBytes)),
			cudaDeleterWrapper<T>);
	};;
}