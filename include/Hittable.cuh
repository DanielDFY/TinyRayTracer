#pragma once

#include <Ray.cuh>
#include <AABB.cuh>
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
		__device__ Hittable(Material* matPtr = nullptr) : _matPtr(matPtr) {}
		__device__ virtual ~Hittable() {};

		__device__ Material* matPtr() const { return _matPtr; }

		__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const = 0;
		__device__ virtual bool boundingBox(float time0, float time1, AABB& outputBox) const = 0;

	protected:
		Material* _matPtr;
	};
}
