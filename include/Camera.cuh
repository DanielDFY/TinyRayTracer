#pragma once

#include <Ray.cuh>

namespace TinyRT {
	class Camera {
	public:
		__device__ Camera(
			const Point3& lookFrom = { 0.0f, 0.0f, 0.0f },
			const Point3& lookAt = { 0.0f, 0.0f, 1.0f },
			const Vec3& vUp = { 0.0f, 1.0f, 0.0f },
			float vFov = 90.0f,
			float aspectRatio = 16.0f / 9.0f,
			float aperture = 0.0f,
			float focusDist = 1.0f,
			float time0 = 0.0f,
			float time1 = 0.0f
		);

		__device__ Ray getRay(float s, float t, curandState* const randStatePtr = nullptr) const;

	private:
		Point3 _origin;
		Point3 _lowerLeftCorner;
		Vec3 _horizontal;
		Vec3 _vertical;
		Vec3 _u, _v, _w;
		float _lensRadius;
		float _time0, _time1;	  // shutter open/close times
	};
}