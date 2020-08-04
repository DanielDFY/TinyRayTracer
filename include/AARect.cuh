#pragma once
#include <Hittable.cuh>

namespace TinyRT {
	class XYRect : public Hittable {
	public:
		__device__ XYRect(float x0, float x1, float y0, float y1, float k, Material* matPtr)
			: _x0(x0), _x1(x1), _y0(y0), _y1(y1), _k(k), _matPtr(matPtr) {
		}

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;
		__device__ bool boundingBox(float t0, float t1, AABB& outputBox) const override {
			// The bounding box must have non-zero width in each dimension, so pad the
			// Z dimension a small amount
			outputBox = AABB({ _x0, _y0, _k - 0.0001f }, { _x1, _y1, _k + 0.0001f });
			return true;
		}

	private:
		Material* _matPtr;
		float _x0, _x1, _y0, _y1, _k;
	};

	class XZRect : public Hittable {
	public:
		__device__ XZRect(float x0, float x1, float z0, float z1, float k, Material* matPtr)
			: _x0(x0), _x1(x1), _z0(z0), _z1(z1), _k(k), _matPtr(matPtr) {
		}

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;
		__device__ bool boundingBox(float t0, float t1, AABB& outputBox) const override {
			// The bounding box must have non-zero width in each dimension, so pad the
			// Y dimension a small amount
			outputBox = AABB({ _x0, _k - 0.0001f, _z0 }, { _x1, _k + 0.0001f, _z1 });
			return true;
		}

	private:
		Material* _matPtr;
		float _x0, _x1, _z0, _z1, _k;
	};

	class YZRect : public Hittable {
	public:
		__device__ YZRect(float y0, float y1, float z0, float z1, float k, Material* matPtr)
			: _y0(y0), _y1(y1), _z0(z0), _z1(z1), _k(k), _matPtr(matPtr) {
		}

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;
		__device__ bool boundingBox(float t0, float t1, AABB& outputBox) const override {
			// The bounding box must have non-zero width in each dimension, so pad the
			// X dimension a small amount
			outputBox = AABB({ _k - 0.0001f, _y0, _z0 }, { _k + 0.0001f, _y1, _z1 });
			return true;
		}

	private:
		Material* _matPtr;
		float _y0, _y1, _z0, _z1, _k;
	};
}
