#include <Material.cuh>

namespace TinyRT {
	__device__ float schlick(float cos, float refIdx) {
		float r0 = (1.0f - refIdx) / (1.0f + refIdx);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * pow((1.0f - cos), 5);
	}

	__device__ bool Lambertian::scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* const randStatePtr) const {
		const Vec3 scatterDirection = rec.normal + randomUnitVec3(randStatePtr);
		scattered = Ray(rec.point, scatterDirection, rayIn.time());
		attenuation = _texturePtr->value(rec.u, rec.v, rec.point);
		return true;
	}

	__device__ bool Metal::scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* const randStatePtr) const {
		const Vec3 reflected = reflect(unitVec3(rayIn.direction()), rec.normal);
		scattered = Ray(rec.point, reflected + _fuzz * randomVec3InUnitSphere(randStatePtr));
		attenuation = _albedo;
		return (dot(scattered.direction(), rec.normal) > 0.0f);
	}

	__device__ bool Dielectric::scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* const randStatePtr) const {
		attenuation = Vec3(1.0f, 1.0f, 1.0f);
		const float etaOverEtaPrime = rec.isFrontFace ? 1.0f / _refIdx : _refIdx;	// 1.0 for air

		const Vec3 unitDirection = unitVec3(rayIn.direction());
		const float cosTheta = fmin(dot(-unitDirection, rec.normal), 1.0f);
		const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
		if (etaOverEtaPrime * sinTheta > 1.0f) {
			// must reflect
			const Vec3 reflected = reflect(unitDirection, rec.normal);
			scattered = Ray(rec.point, reflected);
			return true;
		}
		else {
			// can refract
			const float reflectProb = schlick(cosTheta, etaOverEtaPrime);
			if (curand_uniform(randStatePtr) < reflectProb) {
				const Vec3 reflected = reflect(unitDirection, rec.normal);
				scattered = Ray(rec.point, reflected);
				return true;
			}
			else {
				const Vec3 refracted = refract(unitDirection, rec.normal, etaOverEtaPrime);
				scattered = Ray(rec.point, refracted);
				return true;
			}
		}
	}
}
