#pragma once

#include <Ray.cuh>
#include <cuda_runtime.h>


namespace TinyRT {
	class Material;

	struct HitRecord {
		Point3 point;
		Vec3 normal;
		Material* matPtr;
		double t;
		bool isFrontFace;

		__device__ void setFaceNormal(const Ray& r, const Vec3& outwardNormal) {
			isFrontFace = dot(r.direction(), outwardNormal) < 0;
			normal = isFrontFace ? outwardNormal : -outwardNormal;
		}
	};

	class Hittable {
	public:
		__device__ Hittable() {};
		__device__ Hittable(const Hittable&) {};
		__device__ Hittable(Hittable&&) noexcept {};
		__device__ Hittable& operator=(const Hittable&) { return *this; };
		__device__ Hittable& operator=(Hittable&&) noexcept { return *this; };
		__device__ virtual ~Hittable() {};

		__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const = 0;
	};
}
