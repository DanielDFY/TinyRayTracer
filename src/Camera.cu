#include <Camera.cuh>

namespace TinyRT {
	__device__ Camera::Camera(
		const Point3& lookFrom,
		const Point3& lookAt,
		const Vec3& vUp,
		float vFov,
		float aspectRatio,
		float aperture,
		float focusDist,
		float time0,
		float time1
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
		_time0 = time0;
		_time1 = time1;
	}

	__device__ Ray Camera::getRay(float s, float t, curandState* const randStatePtr) const {
		const Vec3 rd = randStatePtr == nullptr ? Vec3(0.0f, 0.0f, 0.0f) : _lensRadius * randomVec3InUnitDisk(randStatePtr);
		const Vec3 offset = _u * rd.x() + _v * rd.y();

		return {
			_origin + offset,
			_lowerLeftCorner + s * _horizontal + t * _vertical - _origin - offset,
			randomFloat(_time0, _time1, randStatePtr)
		};
	}
}
