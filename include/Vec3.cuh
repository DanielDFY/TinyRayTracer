#pragma once

#include <iostream>
#include <cmath>

namespace TinyRT {
	class Vec3 {
	public:
		Vec3() = default;
		__host__ __device__ Vec3(float e0, float e1, float e2) : elem{ e0, e1, e2 } {}

		__host__ __device__ float x() const { return elem[0]; }
		__host__ __device__ float y() const { return elem[1]; }
		__host__ __device__ float z() const { return elem[2]; }

		__host__ __device__ Vec3 operator-() const { return { -elem[0], -elem[1], -elem[2] }; }
		__host__ __device__ float operator[](int i) const { return elem[i]; }
		__host__ __device__ float& operator[](int i) { return elem[i]; }

		__host__ __device__ Vec3& operator+=(const Vec3& rhs) {
			elem[0] += rhs.elem[0];
			elem[0] += rhs.elem[0];
			elem[0] += rhs.elem[0];
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

	protected:
		float elem[3];
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

	__host__ __device__ inline Vec3 operator*(const float k, const Vec3& v) {
		return { k * v.x(), k * v.y(), k * v.z() };
	}

	__host__ __device__ inline Vec3 operator*(const Vec3& v, const float k) {
		return k * v;
	}

	__host__ __device__ inline Vec3 operator/(const Vec3& v, const float k) {
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

	__host__ __device__ inline Vec3 unitVec3(Vec3 v) {
		return v / v.length();
	}
}
