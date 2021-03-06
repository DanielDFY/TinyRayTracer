#include <Sphere.cuh>

namespace TinyRT {
	__device__ bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const {
		const Vec3 oc = r.origin() - _center;
		const float a = r.direction().lengthSquared();
		const float bHalf = dot(oc, r.direction());
		const float c = oc.lengthSquared() - _radius * _radius;
		const float discriminant = bHalf * bHalf - a * c;

		if (discriminant > 0) {
			const float root = sqrtf(discriminant);
			float t = (-bHalf - root) / a;
			if (t > tMin && t < tMax) {
				rec.t = t;
				rec.point = r.at(t);
				const Vec3 outwardNormal = (rec.point - _center) / _radius;
				rec.setFaceNormal(r, outwardNormal);
				getSphereUV((rec.point - _center) / _radius, rec.u, rec.v);
				rec.matPtr = _matPtr;
				return true;
			}
			t = (-bHalf + root) / a;
			if (t > tMin && t < tMax) {
				rec.t = t;
				rec.point = r.at(t);
				const Vec3 outwardNormal = (rec.point - _center) / _radius;
				rec.setFaceNormal(r, outwardNormal);
				getSphereUV((rec.point - _center) / _radius, rec.u, rec.v);
				rec.matPtr = _matPtr;
				return true;
			}
		}
		return false;
	}

	__device__ bool Sphere::boundingBox(float time0, float time1, AABB& outputBox) const {
		const Vec3 halfVec(_radius, _radius, _radius);

		outputBox = AABB(
			_center - halfVec,
			_center + halfVec
		);

		return true;
	}

}
