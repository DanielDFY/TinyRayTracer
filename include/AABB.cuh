#pragma once

#include <Point3.cuh>
#include <Ray.cuh>

namespace TinyRT {
	class AABB {
	public:
		__host__ __device__ AABB() : _minPoint(Point3(0.0f, 0.0f, 0.0f)), _maxPoint(Point3(0.0f, 0.0f, 0.0f)) {}
		__host__ __device__ AABB(const Point3& minPoint, const Point3& maxPoint)
			: _minPoint(minPoint), _maxPoint(maxPoint) {}

		__host__ __device__ Point3 minPoint() const { return _minPoint; }
		__host__ __device__ Point3 maxPoint() const { return _maxPoint; }

		__device__ bool hit(const Ray& r, float tMin, float tMax) const;

	private:
		Point3 _minPoint;
		Point3 _maxPoint;
	};

	__host__ __device__ AABB surroundingBox(const AABB& box0, const AABB& box1);
}
