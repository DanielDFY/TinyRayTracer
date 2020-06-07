#include <Sphere.cuh>

namespace TinyRT {
	// bug when add -rdc=true to nvcc, so temporarily define in header file
	/*
	__device__ bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const {
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
	*/
}
