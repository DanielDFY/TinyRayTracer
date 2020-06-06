#pragma once

#include <Point3.cuh>
#include <Vec3.cuh>

namespace TinyRT {
	class Ray {
	public:
		__device__ Ray() {}
		__device__ Ray(const Point3& origin, const Vec3& direction)
			: ori(origin), dir(direction) {
		}

		__device__ Point3 origin() const { return ori; }
		__device__ Vec3 direction() const { return dir; }

		__device__ Point3 at(float t) const { return ori + t * dir; }

	private:
		Point3 ori;
		Vec3 dir;	// may not be unit length
	};
}
