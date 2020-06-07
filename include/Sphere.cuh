#pragma once

#include <Hittable.cuh>

namespace TinyRT {
	class Sphere : public Hittable {
	public:
		__device__ Sphere() = delete;
		__device__ Sphere(Point3 center, float radius) : Hittable(), _center(center), _radius(radius) {}

		__device__ Point3 center() const { return _center; }
		__device__ float radius() const { return _radius; }

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override {
			const Vec3 oc = r.origin() - _center;
			const float a = r.direction().lengthSquared();
			const float bHalf = dot(oc, r.direction());
			const float c = oc.lengthSquared() - radius() * radius();
			const float discriminant = bHalf * bHalf - a * c;

			if (discriminant > 0) {
				const float root = sqrt(discriminant);
				float t = (-bHalf - root) / a;
				if (t > tMin && t < tMax) {
					rec.t = t;
					rec.point = r.at(t);
					const Vec3 outwardNormal = (rec.point - _center) / _radius;
					rec.setFaceNormal(r, outwardNormal);
					return true;
				}
				t = (-bHalf + root) / a;
				if (t > tMin && t < tMax) {
					rec.t = t;
					const Vec3 outwardNormal = (rec.point - _center) / _radius;
					rec.setFaceNormal(r, outwardNormal);
					return true;
				}
			}

			return false;
		}

	private:
		Point3 _center;
		float _radius;
	};
}
