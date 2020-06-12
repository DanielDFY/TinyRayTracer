#pragma once

#include <Ray.cuh>
#include <Hittable.cuh>
#include <Color.cuh>

namespace TinyRT {
	__device__ float schlick(float cos, float refIdx) {
		float r0 = (1.0f - refIdx) / (1.0f + refIdx);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * pow((1.0f - cos), 5);
	}
	
	class Material {
	public:
		__device__ Material() {};
		__device__ Material(const Material&) {};
		__device__ Material(Material&&) noexcept {};
		__device__ Material& operator=(const Material&) { return *this; };
		__device__ Material& operator=(Material&&) noexcept { return *this; };
		__device__ virtual ~Material() {};

		__device__ virtual bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* randStatePtr) const = 0;
	};

	// bug when add -rdc=true to nvcc and separate into .cu and .cuh, so define in header file

	class Lambertian : public Material {
	public:
		__device__ Lambertian(const Color& albedo) : _albedo(albedo) {}

		__device__ bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* const randStatePtr) const override {
			const Vec3 scatterDirection = rec.normal + randomUnitVec3(randStatePtr);
			scattered = Ray(rec.point, scatterDirection);
			attenuation = _albedo;
			return true;
		}

	private:
		Color _albedo;
	};

	class Metal : public Material {
	public:
		__device__ Metal(const Color& albedo, float fuzz) : _albedo(albedo), _fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

		__device__ bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* const randStatePtr) const override {
			const Vec3 reflected = reflect(unitVec3(rayIn.direction()), rec.normal);
			scattered = Ray(rec.point, reflected + _fuzz * randomVec3InUnitSphere(randStatePtr));
			attenuation = _albedo;
			return (dot(scattered.direction(), rec.normal) > 0.0f);
		}

	private:
		Color _albedo;
		float _fuzz;
	};

	class Dielectric : public Material {
	public:
		__device__ Dielectric(float refIdx) : _refIdx(refIdx) {}

		__device__ bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* const randStatePtr) const override {
			attenuation = Vec3(1.0f, 1.0f, 1.0f);
			const float etaOverEtaPrime = rec.isFrontFace ? 1.0f / _refIdx : _refIdx;	// 1.0 for air

			const Vec3 unitDirection = unitVec3(rayIn.direction());
			const float cosTheta = fmin(dot(-unitDirection, rec.normal), 1.0f);
			const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
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
				} else {
					const Vec3 refracted = refract(unitDirection, rec.normal, etaOverEtaPrime);
					scattered = Ray(rec.point, refracted);
					return true;
				}
			}
		}

	private:
		float _refIdx;
	};
}
