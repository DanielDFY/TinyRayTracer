#pragma once

#include <Ray.cuh>

namespace TinyRT {
	class Camera {
	public:
		__device__ Camera() {
			const float aspectRatio = 16.0f / 9.0f;
			const float viewportHeight = 2.0f;
			const float viewportWidth = aspectRatio * viewportHeight;
			const float focalLength = 1.0f;

			_origin = Point3(0.0f, 0.0f, 0.0f);
			_horizontal = Vec3(viewportWidth, 0.0f, 0.0f);
			_vertical = Vec3(0.0f, viewportHeight, 0.0f);
			// left-handed Y up
			_lowerLeftCorner = _origin - _horizontal / 2.0f - _vertical / 2.0f + Vec3(0.0f, 0.0f, focalLength);
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
