#pragma once

#include <Ray.cuh>
#include <Hittable.cuh>
#include <Color.cuh>
#include <Texture.cuh>

namespace TinyRT {
	__device__ float schlick(float cos, float refIdx);
	
	class Material {
	public:
		__device__ Material(Texture* texturePtr = nullptr) : _texturePtr(texturePtr) {}

		__device__ Texture* texturePtr() const { return _texturePtr; }

		__device__ virtual bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* randStatePtr) const = 0;
		__device__ virtual Color emitted(float u, float v, const Point3& point) const {
			return { 0.0f, 0.0f, 0.0f };
		}

	protected:
		Texture* _texturePtr;
	};

	class Lambertian : public Material {
	public:
		__device__ Lambertian(Texture* texturePtr) : Material(texturePtr) {}

		__device__ bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* randStatePtr) const override;
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

	class DiffuseLight : public Material {
	public:
		__device__ DiffuseLight(Texture* emitTex) : _emitTex(emitTex) {}

		__device__ bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* randStatePtr) const override {
			return false;
		}

		__device__ Color emitted(float u, float v, const Point3& point) const override {
			return _emitTex->value(u, v, point);
		}

	private:
		Texture* _emitTex;
	};
}
