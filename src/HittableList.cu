#include <HittableList.cuh>
#include <AABB.cuh>

namespace TinyRT {
	__device__ bool HittableList::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const {
		HitRecord tempRec;
		bool isHit = false;
		float closest = tMax;

		for (size_t i = 0; i < _size; ++i) {
			if (_ptrList[i]->hit(r, tMin, closest, tempRec)) {
				isHit = true;
				closest = tempRec.t;
				rec = tempRec;
			}
		}

		return isHit;
	}

	__device__ bool HittableList::boundingBox(float time0, float time1, AABB& outputBox) const {
		if (_size == 0) return false;

		AABB tempBox;
		bool isFirstBox = true;

		for (size_t i = 0; i < _size; ++i) {
			if (!_ptrList[i]->boundingBox(time0, time1, tempBox)) return false;
			outputBox = isFirstBox ? tempBox : surroundingBox(outputBox, tempBox);
			isFirstBox = false;
		}

		return true;
	}

}
