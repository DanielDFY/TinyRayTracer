#pragma once

#include <iostream>
#include <cmath>

namespace TinyRT {
	class Vec3 {
	public:
		__host__ __device__ Vec3() : elem{0.0f, 0.0f, 0.0f} {};
		__host__ __device__ Vec3(float e0, float e1, float e2) : elem{ e0, e1, e2 } {}

		__host__ __device__ float x() const { return elem[0]; }
		__host__ __device__ float y() const { return elem[1]; }
		__host__ __device__ float z() const { return elem[2]; }

		__host__ __device__ Vec3 operator-() const { return { -elem[0], -elem[1], -elem[2] }; }
		__host__ __device__ float operator[](int i) const { return elem[i]; }
		__host__ __device__ float& operator[](int i) { return elem[i]; }

		__host__ __device__ Vec3& operator+=(const Vec3& v) {
			elem[0] += v.elem[0];
			elem[0] += v.elem[0];
			elem[0] += v.elem[0];
			return *this;
		}

		__host__ __device__ Vec3& operator*=(const float k) {
			elem[0] *= k;
			elem[1] *= k;
			elem[2] *= k;
			return *this;
		}

		__host__ __device__ Vec3& operator/=(const float k) {
			return *this *= 1 / k;
		}

		__host__ __device__ float lengthSquared() const {
			return elem[0] * elem[0] + elem[1] * elem[1] + elem[2] * elem[2];
		}

		__host__ __device__ float length() const {
			return sqrt(lengthSquared());
		}

	public:
		float elem[3];
	};

	// Utility functions
	inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
		return out << v.elem[0] << ' ' << v.elem[1] << ' ' << v.elem[2];
	}

	__host__ __device__ inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs) {
		return { lhs.elem[0] + rhs.elem[0], lhs.elem[1] + rhs.elem[1], lhs.elem[2] + rhs.elem[2] };
	}

	__host__ __device__ inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs) {
		return { lhs.elem[0] - rhs.elem[0], lhs.elem[1] - rhs.elem[1], lhs.elem[2] - rhs.elem[2] };
	}

	__host__ __device__ inline Vec3 operator*(const Vec3& lhs, const Vec3& rhs) {
		return { lhs.elem[0] * rhs.elem[0], lhs.elem[1] * rhs.elem[1], lhs.elem[2] * rhs.elem[2] };
	}

	__host__ __device__ inline Vec3 operator*(const Vec3& v, const float k) {
		return v * k;
	}

	__host__ __device__ inline Vec3 operator/(const Vec3& v, const float k) {
		return v * (1 / k);
	}

	__host__ __device__ inline float dot(const Vec3& lhs, const Vec3& rhs) {
		return lhs.elem[0] * rhs.elem[0] + lhs.elem[1] * rhs.elem[1] + lhs.elem[2] * rhs.elem[2];
	}

	__host__ __device__ inline float cross(const Vec3& lhs, const Vec3& rhs) {
		return lhs.elem[1] * rhs.elem[2] - lhs.elem[2] * rhs.elem[1]
			+ lhs.elem[2] * rhs.elem[0] - lhs.elem[0] * rhs.elem[2]
			+ lhs.elem[0] * rhs.elem[1] - lhs.elem[1] * rhs.elem[0];
	}

	__host__ __device__ inline Vec3 unitVec3(Vec3 v) {
		return v / v.length();
	}
}
