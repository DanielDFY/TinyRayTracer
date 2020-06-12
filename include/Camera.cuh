#pragma once

#include <Ray.cuh>

namespace TinyRT {
	class Camera {
	public:
		// bug when add -rdc=true to nvcc and separate into .cu and .cuh, so define in header file
		__device__ Camera(
			const Point3& lookFrom = { 0.0f, 0.0f, 0.0f },
			const Point3& lookAt = { 0.0f, 0.0f, 1.0f },
			const Vec3& vUp = { 0.0f, 1.0f, 0.0f },
			float vFov = 90.0f,
			float aspectRatio = 16.0f / 9.0f,
			float aperture = 0.0f,
			float focusDist = 1.0f
		) {
			const float theta = degreeToRadian(vFov);
			const float h = tan(theta / 2.0f);
			const float viewportHeight = 2.0f * h;
			const float viewportWidth = aspectRatio * viewportHeight;

			_w = unitVec3(lookAt - lookFrom);
			_u = unitVec3(cross(vUp, _w));
			_v = cross(_w, _u);

			_origin = lookFrom;
			_horizontal = focusDist * viewportWidth * _u;
			_vertical = focusDist * viewportHeight * _v;
			_lowerLeftCorner = _origin - _horizontal / 2 - _vertical / 2 + focusDist * _w;
			_lensRadius = aperture / 2;
		}

		__device__ Ray getRay(float s, float t, curandState* const randStatePtr = nullptr) const {
			const Vec3 rd = randStatePtr == nullptr ? Vec3(0.0f, 0.0f, 0.0f) : _lensRadius * randomVec3InUnitDisk(randStatePtr);
			const Vec3 offset = _u * rd.x() + _v * rd.y();

			return { _origin + offset, _lowerLeftCorner + s * _horizontal + t * _vertical - _origin - offset };
		}

	private:
		Point3 _origin;
		Point3 _lowerLeftCorner;
		Vec3 _horizontal;
		Vec3 _vertical;
		Vec3 _u, _v, _w;
		float _lensRadius;
	};
}