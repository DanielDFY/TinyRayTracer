#pragma once

#include <iostream>
#include <cmath>

#include <helperUtils.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

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
			return sqrt(lengthSquared());
		}

		inline static Vec3 random() {
			return Vec3(randomFloat(), randomFloat(), randomFloat());
		}

		inline static Vec3 random(float min, float max) {
			return Vec3(randomFloat(min, max), randomFloat(min, max), randomFloat(min, max));
		}

		__device__ inline static Vec3 random(curandState* const randStatePtr) {
			return Vec3(curand_uniform(randStatePtr), curand_uniform(randStatePtr), curand_uniform(randStatePtr));
		}

		__device__ inline static Vec3 random(curandState* const randStatePtr, float min, float max) {
			return Vec3(
				min + (max - min) * curand_uniform(randStatePtr),
				min + (max - min) * curand_uniform(randStatePtr),
				min + (max - min) * curand_uniform(randStatePtr)
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

	__host__ __device__ inline float cross(const Vec3& lhs, const Vec3& rhs) {
		return lhs.y() * rhs.z() - lhs.z() * rhs.y()
			+ lhs.z() * rhs.x() - lhs.x() * rhs.z()
			+ lhs.x() * rhs.y() - lhs.y() * rhs.x();
	}

	__host__ __device__ inline Vec3 unitVec3(const Vec3 v) {
		return v / v.length();
	}

	// bug when add -rdc=true to nvcc and separate into .cu and .cuh, so define in header file

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
		const auto r = sqrt(1 - z * z);
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
			const Vec3 p = Vec3::random(randStatePtr, -1.0f, 1.0f);
			if (p.lengthSquared() >= 1) continue;
			return p;
		}
	}

	__device__ Vec3 randomUnitVec3(curandState* const randStatePtr) {
		const auto a = 2.0f * M_PI * curand_uniform(randStatePtr);
		const auto z = -1.0f + 2.0f * curand_uniform(randStatePtr);
		const auto r = sqrt(1 - z * z);
		return { r * cos(a), r * sin(a), z };
	}

	__device__ Vec3 randomVec3InHemisphere(const Vec3& normal, curandState* const randStatePtr) {
		const Vec3 vec3InUnitSphere = randomVec3InUnitSphere(randStatePtr);
		if (dot(vec3InUnitSphere, normal) > 0.0f) // In the same hemisphere as the normal
			return vec3InUnitSphere;
		else
			return -vec3InUnitSphere;
	}

	__host__ __device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
		return v - 2.0f * dot(v, n) * n;
	}
}
