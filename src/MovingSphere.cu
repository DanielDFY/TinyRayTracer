#include <MovingSphere.cuh>

namespace TinyRT {
	__device__ bool MovingSphere::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const {
		const Vec3 oc = r.origin() - center(r.time());
		const float a = r.direction().lengthSquared();
		const float bHalf = dot(oc, r.direction());
		const float c = oc.lengthSquared() - _radius * _radius;

		const float discriminant = bHalf * bHalf - a * c;

		if (discriminant > 0.0f) {
			const float root = sqrtf(discriminant);

			float temp = (-bHalf - root) / a;
			if (temp > tMin && temp < tMax) {
				rec.t = temp;
				rec.point = r.at(rec.t);
				const Vec3 outwardNormal = (rec.point - center(r.time())) / _radius;
				rec.setFaceNormal(r, outwardNormal);
				rec.matPtr = _matPtr;
				return true;
			}

			temp = (-bHalf + root) / a;
			if (temp < tMax && temp > tMin) {
				rec.t = temp;
				rec.point = r.at(rec.t);
				const Vec3 outwardNormal = (rec.point - center(r.time())) / _radius;
				rec.setFaceNormal(r, outwardNormal);
				rec.matPtr = _matPtr;
				return true;
			}
		}
		return false;
	}
}
