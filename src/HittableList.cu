#include <HittableList.cuh>

namespace TinyRT {
	__device__ bool HittableList::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const {
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
}
