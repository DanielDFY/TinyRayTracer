#pragma once

#include <Hittable.cuh>

namespace TinyRT {
	class HittableList : public Hittable {
	public:
		__device__ HittableList() {};
		__device__ HittableList(Hittable** const list, const size_t size) : _list(list), _size(size) {}

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override {
			HitRecord tempRec;
			bool isHit = false;
			float closest = tMax;

			for (size_t i = 0; i < _size; ++i) {
				if (_list[i]->hit(r, tMin, closest, tempRec)) {
					isHit = true;
					closest = tempRec.t;
					rec = tempRec;
				}
			}

			return isHit;
		}

	private:
		Hittable** _list;
		size_t _size;
	};
}
