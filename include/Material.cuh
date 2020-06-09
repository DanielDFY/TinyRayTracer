#pragma once

#include <Ray.cuh>
#include <Hittable.cuh>
#include <Color.cuh>

namespace TinyRT {
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
}
