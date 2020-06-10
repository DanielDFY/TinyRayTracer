#pragma once

#include <Ray.cuh>

namespace TinyRT {
	class Camera {
	public:
		__device__ Camera() = delete;
		__device__ Camera(const Point3& lookFrom, const Point3& lookAt, const Vec3& vUp, float vFov, float aspectRatio) {
			const float theta = degreeToRadian(vFov);
			const float h = tan(theta / 2.0f);
			const float viewportHeight = 2.0f * h;
			const float viewportWidth = aspectRatio * viewportHeight;

			const Vec3 w = unitVec3(lookAt - lookFrom);
			const Vec3 u = unitVec3(cross(vUp, w));
			const Vec3 v = cross(w, u);

			_origin = lookFrom;
			_horizontal = viewportWidth * u;
			_vertical = viewportHeight * v;
			_lowerLeftCorner = _origin - _horizontal / 2 - _vertical / 2 + w;
		}

		__device__ Ray getRay(float u, float v) const {
			return { _origin, _lowerLeftCorner + u * _horizontal + v * _vertical - _origin };
		}

	private:
		Point3 _origin;
		Point3 _lowerLeftCorner;
		Vec3 _horizontal;
		Vec3 _vertical;
	};
}
