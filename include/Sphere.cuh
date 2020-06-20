#pragma once

#include <Hittable.cuh>

namespace TinyRT {
	class Sphere : public Hittable {
	public:
		__device__ Sphere() = delete;
		__device__ Sphere(Point3 center, float radius, Material* matPtr = nullptr)
			: Hittable(matPtr), _center(center), _radius(radius) {}

		__device__ Point3 center() const { return _center; }
		__device__ float radius() const { return _radius; }

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;

	private:
		Point3 _center;
		float _radius;
	};
}
