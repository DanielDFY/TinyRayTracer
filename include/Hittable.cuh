#pragma once

#include <Ray.cuh>
#include <cuda_runtime.h>

namespace TinyRT {
	struct HitRecord {
		Point3 point;
		Vec3 normal;
		double t;
		bool isFrontFace;

		__device__ void setFaceNormal(const Ray& r, const Vec3& outwardNormal) {
			isFrontFace = dot(r.direction(), outwardNormal) < 0;
			normal = isFrontFace ? outwardNormal : -outwardNormal;
		}
	};

	class Hittable {
	public:
		__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const = 0;
	};
}
