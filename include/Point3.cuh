#pragma once

#include <Vec3.cuh>

namespace TinyRT {
	class Point3 : public Vec3 {
	public:
		__host__ __device__ Point3() : Vec3() {};
		__host__ __device__ Point3(float e0, float e1, float e2) : Vec3(e0, e1, e2) {};
	};
}
