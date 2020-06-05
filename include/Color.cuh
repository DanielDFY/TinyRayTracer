#pragma once

#include <Vec3.cuh>

namespace TinyRT {
	class Color : public Vec3 {
	public:
		__host__ __device__ Color() : Vec3() {};
		__host__ __device__ Color(float e0, float e1, float e2) : Vec3(e0, e1, e2) {};

		__host__ __device__ float r() const { return elem[0]; }
		__host__ __device__ float g() const { return elem[1]; }
		__host__ __device__ float b() const { return elem[2]; }

		__host__ __device__ int r8bit() const { return static_cast<int>(255 * elem[0]); }
		__host__ __device__ int g8bit() const { return static_cast<int>(255 * elem[1]); }
		__host__ __device__ int b8bit() const { return static_cast<int>(255 * elem[2]); }
	};
}
