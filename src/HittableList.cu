#include <HittableList.cuh>

namespace TinyRT {
	// bug when add -rdc=true to nvcc, so temporarily define in header file
	/*
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
	*/
}
