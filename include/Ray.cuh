#pragma once

#include <Point3.cuh>
#include <Vec3.cuh>

namespace TinyRT {
	class Ray {
	public:
		__device__ Ray() {}
		__device__ Ray(const Point3& origin, const Vec3& direction)
			: _origin(origin), _direction(direction) {
		}

		__device__ Point3 origin() const { return _origin; }
		__device__ Vec3 direction() const { return _direction; }

		__device__ Point3 at(float t) const { return _origin + t * _direction; }

	private:
		Point3 _origin;
		Vec3 _direction;	// may not be unit length
	};
}
