#pragma once

#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>

namespace TinyRT {
	/* Math Utils */
	constexpr double DOUBLE_INFINITY = std::numeric_limits<float>::infinity();
	constexpr double PI = 3.1415926535897932385;

	inline double degreeToRadian(const double degree) { return degree * PI / 180.0; }
	
	/* Error Checking */
	#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

	template<typename T>
	void check(T err, const char* func, const char* file, int line) {
		if (err != cudaSuccess) {
			std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
			std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
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