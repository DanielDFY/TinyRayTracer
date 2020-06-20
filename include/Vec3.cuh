#pragma once

#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <helperUtils.cuh>

namespace TinyRT {
	class Vec3 {
	public:
		Vec3() = default;
		__host__ __device__ Vec3(float e0, float e1, float e2) : _elem{ e0, e1, e2 } {}

		__host__ __device__ float x() const { return _elem[0]; }
		__host__ __device__ float y() const { return _elem[1]; }
		__host__ __device__ float z() const { return _elem[2]; }

		__host__ __device__ Vec3 operator-() const { return { -_elem[0], -_elem[1], -_elem[2] }; }
		__host__ __device__ float operator[](int i) const { return _elem[i]; }
		__host__ __device__ float& operator[](int i) { return _elem[i]; }

		__host__ __device__ Vec3& operator+=(const Vec3& rhs) {
			_elem[0] += rhs._elem[0];
			_elem[1] += rhs._elem[1];
			_elem[2] += rhs._elem[2];
			return *this;
		}

		__host__ __device__ Vec3& operator*=(const Vec3& rhs) {
			_elem[0] *= rhs._elem[0];
			_elem[1] *= rhs._elem[1];
			_elem[2] *= rhs._elem[2];
			return *this;
		}

		__host__ __device__ Vec3& operator*=(float k) {
			_elem[0] *= k;
			_elem[1] *= k;
			_elem[2] *= k;
			return *this;
		}

		__host__ __device__ Vec3& operator/=(float k) {
			return *this *= 1 / k;
		}

		__host__ __device__ float lengthSquared() const {
			return _elem[0] * _elem[0] + _elem[1] * _elem[1] + _elem[2] * _elem[2];
		}

		__host__ __device__ float length() const {
			return sqrtf(lengthSquared());
		}

		static Vec3 random() {
			return Vec3(randomFloat(), randomFloat(), randomFloat());
		}

		static Vec3 random(float min, float max) {
			return Vec3(randomFloat(min, max), randomFloat(min, max), randomFloat(min, max));
		}

		__device__ static Vec3 random(curandState* const randStatePtr) {
			return randStatePtr == nullptr ? Vec3(0.0f, 0.0f, 0.0f) : Vec3(randomFloat(randStatePtr), randomFloat(randStatePtr), randomFloat(randStatePtr));
		}

		__device__ static Vec3 random(float min, float max, curandState* const randStatePtr) {
			return randStatePtr == nullptr ? Vec3(0.0f, 0.0f, 0.0f) : 
				Vec3(
				randomFloat(min, max, randStatePtr),
				randomFloat(min, max, randStatePtr),
				randomFloat(min, max, randStatePtr)
			);
		}

	protected:
		float _elem[3];
	};

	// Utility functions
	inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
		return out << v.x() << ' ' << v.y() << ' ' << v.z();
	}

	__host__ __device__ inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs) {
		return { lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z() };
	}

	__host__ __device__ inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs) {
		return { lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z() };
	}

	__host__ __device__ inline Vec3 operator*(const Vec3& lhs, const Vec3& rhs) {
		return { lhs.x() * rhs.x(), lhs.y() * rhs.y(), lhs.z() * rhs.z() };
	}

	__host__ __device__ inline Vec3 operator*(float k, const Vec3& v) {
		return { k * v.x(), k * v.y(), k * v.z() };
	}

	__host__ __device__ inline Vec3 operator*(const Vec3& v, float k) {
		return k * v;
	}

	__host__ __device__ inline Vec3 operator/(const Vec3& v, float k) {
		return (1 / k) * v;
	}

	__host__ __device__ inline float dot(const Vec3& lhs, const Vec3& rhs) {
		return lhs.x() * rhs.x() + lhs.y() * rhs.y() + lhs.z() * rhs.z();
	}

	__host__ __device__ inline Vec3 cross(const Vec3& lhs, const Vec3& rhs) {
		return { lhs.y() * rhs.z() - lhs.z() * rhs.y(),
			lhs.z() * rhs.x() - lhs.x() * rhs.z(),
			lhs.x() * rhs.y() - lhs.y() * rhs.x() };
	}

	__host__ __device__ inline Vec3 unitVec3(const Vec3 v) {
		return v / v.length();
	}

	Vec3 randomVec3InUnitSphere();

	Vec3 randomUnitVec3();

	Vec3 randomVec3InHemisphere(const Vec3& normal);

	__device__ Vec3 randomVec3InUnitSphere(curandState* randStatePtr);

	__device__ Vec3 randomUnitVec3(curandState* randStatePtr);

	__device__ Vec3 randomVec3InHemisphere(const Vec3& normal, curandState* randStatePtr);

	__device__ Vec3 randomVec3InUnitDisk(curandState* randStatePtr);

	__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
		return v - 2.0f * dot(v, n) * n;
	}

	__host__ __device__ Vec3 refract(const Vec3& uv, const Vec3& n, float etaOverEtaPrime);
}
