#pragma once

#include <Hittable.cuh>

namespace TinyRT {
	class HittableList : public Hittable {
	public:
		__device__ HittableList() {};
		__device__ HittableList(Hittable** const list, size_t size) : _list(list), _size(size) {}

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;

	private:
		Hittable** _list;
		size_t _size;
	};
}
