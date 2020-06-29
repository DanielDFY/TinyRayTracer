#pragma once

#include <Color.cuh>
#include <Point3.cuh>

#include <Perlin.cuh>

namespace TinyRT {
	class Texture {
	public:
		__device__ virtual Color value(float u, float v, const Point3& point) const = 0;
	};

	class SolidColor : public Texture {
	public:
		__device__ SolidColor(const Color& color) : _colorValue(color) {}
		__device__ SolidColor(float red, float green, float blue)
			: SolidColor(Color(red, green, blue)) {}

		__device__ Color value(float u, float v, const Point3& point) const override {
			return _colorValue;
		}

	private:
		Color _colorValue;
	};

	class CheckerTexture : public Texture {
	public:
		__device__ CheckerTexture(Texture* oddTexture, Texture* evenTexture)
			: _oddTexture(oddTexture), _evenTexture(evenTexture) {}

		__device__ Color value(float u, float v, const Point3& point) const override;

	private:
		Texture* _oddTexture;
		Texture* _evenTexture;
	};

	class NoiseTexture : public Texture {
	public:
		__device__ NoiseTexture(curandState* randStatePtr) : _noise(randStatePtr) {}
		__device__ NoiseTexture(float scale , curandState* randStatePtr) : _scale(scale), _noise(randStatePtr) {}

		__device__ Color value(float u, float v, const Point3& p) const override {
			return Color(1.0f, 1.0f, 1.0f) * 0.5f * (1.0f + sin(_scale * p.z() + 10.0f * _noise.turb(p)));
		}

	private:
		Perlin _noise;
		float _scale;
	};
}
