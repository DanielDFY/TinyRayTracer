#pragma once

#include <iostream>
#include <cuda_runtime_api.h>

namespace TinyRT {
	#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

	template<typename T>
	void check(T err, const char* const func, const char* const file, const int line) {
		if (err != cudaSuccess) {
			std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
			std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
			exit(EXIT_FAILURE);
		}
	}
}