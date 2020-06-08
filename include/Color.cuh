#pragma once

#include <Vec3.cuh>

namespace TinyRT {
	class Color : public Vec3 {
	public:
		Color() = default;
		__host__ __device__ Color(float e0, float e1, float e2) : Vec3(e0, e1, e2) {};
		__host__ __device__ Color(Vec3 v) : Color(v.x(), v.y(), v.z()) {}

		__host__ __device__ void clamp(int samplesPerPixel) {
			const float scale = 1.0f / static_cast<float>(samplesPerPixel);
			_elem[0] *= scale;
			_elem[1] *= scale;
			_elem[2] *= scale;
		}

		__host__ __device__ float r() const { return _elem[0]; }
		__host__ __device__ float g() const { return _elem[1]; }
		__host__ __device__ float b() const { return _elem[2]; }

		__host__ __device__ int r8bit() const { return static_cast<int>(255 * _elem[0]); }
		__host__ __device__ int g8bit() const { return static_cast<int>(255 * _elem[1]); }
		__host__ __device__ int b8bit() const { return static_cast<int>(255 * _elem[2]); }
	};
}
