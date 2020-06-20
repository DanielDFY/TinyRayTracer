#include <Vec3.cuh>

namespace TinyRT {
	Vec3 randomVec3InUnitSphere() {
		while (true) {
			const Vec3 p = Vec3::random(-1.0f, 1.0f);
			if (p.lengthSquared() >= 1) continue;
			return p;
		}
	}

	Vec3 randomUnitVec3() {
		const auto a = randomFloat(0.0f, 2 * M_PI);
		const auto z = randomFloat(-1.0f, 1.0f);
		const auto r = sqrtf(1 - z * z);
		return { r * cos(a), r * sin(a), z };
	}

	Vec3 randomVec3InHemisphere(const Vec3& normal) {
		const Vec3 vec3InUnitSphere = randomVec3InUnitSphere();
		if (dot(vec3InUnitSphere, normal) > 0.0f) // In the same hemisphere as the normal
			return vec3InUnitSphere;
		else
			return -vec3InUnitSphere;
	}
	
	__device__ Vec3 randomVec3InUnitSphere(curandState* const randStatePtr) {
		while (true) {
			const Vec3 p = randStatePtr == nullptr ? Vec3(0.0f, 0.0f, 0.0f) : Vec3::random(-1.0f, 1.0f, randStatePtr);
			if (p.lengthSquared() >= 1) continue;
			return p;
		}
	}

	__device__ Vec3 randomUnitVec3(curandState* const randStatePtr) {
		if (randStatePtr == nullptr)
			return { 0.0f, 0.0f, 0.0f };
		const auto a = 2.0f * M_PI * randomFloat(randStatePtr);
		const auto z = -1.0f + 2.0f * randomFloat(randStatePtr);
		const auto r = sqrtf(1 - z * z);
		return { r * cos(a), r * sin(a), z };
	}

	__device__ Vec3 randomVec3InHemisphere(const Vec3& normal, curandState* const randStatePtr) {
		const Vec3 vec3InUnitSphere = randStatePtr == nullptr ? Vec3(0.0f, 0.0f, 0.0f) : randomVec3InUnitSphere(randStatePtr);
		if (dot(vec3InUnitSphere, normal) > 0.0f) // In the same hemisphere as the normal
			return vec3InUnitSphere;
		else
			return -vec3InUnitSphere;
	}

	__device__ Vec3 randomVec3InUnitDisk(curandState* const randStatePtr) {
		while (true) {
			const auto v = randStatePtr == nullptr ? Vec3(0.0f, 0.0f, 0.0f) :
				Vec3(-1.0f + randomFloat(randStatePtr), -1.0f + randomFloat(randStatePtr), 0.0f);
			if (v.lengthSquared() >= 1.0f) continue;
			return v;
		}
	}

	__host__ __device__ Vec3 refract(const Vec3& uv, const Vec3& n, float etaOverEtaPrime) {
		const float cosTheta = dot(-uv, n);
		const Vec3 rayOutParallel = etaOverEtaPrime * (uv + cosTheta * n);
		const Vec3 rayOutPerpendicular = -sqrtf(1.0f - rayOutParallel.lengthSquared()) * n;
		return rayOutParallel + rayOutPerpendicular;
	}
}
