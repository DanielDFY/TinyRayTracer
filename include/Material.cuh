#pragma once

#include <Ray.cuh>
#include <Hittable.cuh>
#include <Color.cuh>

namespace TinyRT {
	__device__ float schlick(float cos, float refIdx);
	
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

	class Lambertian : public Material {
	public:
		__device__ Lambertian(const Color& albedo) : _albedo(albedo) {}

		__device__ bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* randStatePtr) const override;

	private:
		Color _albedo;
	};

	class Metal : public Material {
	public:
		__device__ Metal(const Color& albedo, float fuzz) : _albedo(albedo), _fuzz(fuzz < 1.0f ? fuzz : 1.0f) {}

		__device__ bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* randStatePtr) const override;

	private:
		Color _albedo;
		float _fuzz;
	};

	class Dielectric : public Material {
	public:
		__device__ Dielectric(float refIdx) : _refIdx(refIdx) {}

		__device__ bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* randStatePtr) const override;

	private:
		float _refIdx;
	};
}
