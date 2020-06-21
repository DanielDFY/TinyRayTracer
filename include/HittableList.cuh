#pragma once

#include <Hittable.cuh>

namespace TinyRT {
	class HittableList : public Hittable {
	public:
		__device__ HittableList() : _ptrList(nullptr), _size(0) {}
		__device__ HittableList(Hittable** const ptrList, size_t size) : _ptrList(ptrList), _size(size) {}

		__device__ Hittable** ptrList() const { return _ptrList; }
		__device__ size_t size() const { return _size; }

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;
		__device__ bool boundingBox(float time0, float time1, AABB& outputBox) const override;

	private:
		Hittable** _ptrList;
		size_t _size;
	};
}
