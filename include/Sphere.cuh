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
		__device__ bool boundingBox(float time0, float time1, AABB& outputBox) const override;

	private:
		Point3 _center;
		float _radius;
	};

	__device__ inline void getSphereUV(const Point3& point, float& u, float& v) {
		const float phi = atan2(point.z(), point.x());
		const float theta = asin(point.y());
		u = 0.5f - phi / (2 * M_PI);		// 1-(phi + M_PI) / (2*M_PI);
		v = theta / M_PI + 0.5f;			// (theta + M_PI/2) / M_PI;
	}
}
