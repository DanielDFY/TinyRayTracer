#pragma once

#include <Hittable.cuh>

namespace TinyRT {
	class MovingSphere : public Hittable {
	public:
		__device__ MovingSphere() = delete;
		__device__ MovingSphere(
			Point3 center0,
			Point3 center1,
			float time0,
			float time1,
			float radius,
			Material* matPtr
		) : Hittable(matPtr), _center0(center0), _center1(center1),
		    _time0(time0), _time1(time1), _radius(radius) { }

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;
		__device__ bool boundingBox(float time0, float time1, AABB& outputBox) const override;

		__device__ Point3 center(float time) const {
			return _center0 + (time - _time0) / (_time1 - _time0) * (_center1 - _center0);
		}

	private:
		Point3 _center0, _center1;
		float _time0, _time1;
		float _radius;
	};
}
