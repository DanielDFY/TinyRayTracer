#pragma once

#include <Point3.cuh>

namespace TinyRT {
	class Perlin {
	public:
		__device__ Perlin(curandState* randStatePtr);
		__device__ ~Perlin();

		__device__ float noise(const Point3& p) const;
		__device__ float turb(const Point3& p, int depth = 7) const;

	private:
		static const int _pointCount = 256;
		Vec3* _ranVecList;
		int* _permX;
		int* _permY;
		int* _permZ;

		__device__ static int* perlinGeneratePerm(curandState* randStatePtr);

		__device__ static void permute(int* p, int n, curandState* randStatePtr);

		__device__ inline static float perlinInterp(Vec3 c[2][2][2], float u, float v, float w);
	};
}