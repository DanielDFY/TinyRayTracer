#include <AABB.cuh>

namespace TinyRT {
	__device__ bool AABB::hit(const Ray& r, float tMin, float tMax) const {
        for (int i = 0; i < 3; ++i) {
            const float invD = 1.0f / r.direction()[i];
            float t0 = (_minPoint[i] - r.origin()[i]) * invD;
            float t1 = (_maxPoint[i] - r.origin()[i]) * invD;
            if (invD < 0.0f) {
                const float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            tMin = t0 > tMin ? t0 : tMin;
            tMax = t1 < tMax ? t1 : tMax;
            if (tMax <= tMin)
                return false;
        }
        return true;
	}

    __host__ __device__ AABB surroundingBox(const AABB& box0, const AABB& box1) {
        const Point3 smallPoint(fmin(box0.minPoint().x(), box1.minPoint().x()),
            fmin(box0.minPoint().y(), box1.minPoint().y()),
            fmin(box0.minPoint().z(), box1.minPoint().z()));

        const Point3 bigPoint(fmax(box0.maxPoint().x(), box1.maxPoint().x()),
            fmax(box0.maxPoint().y(), box1.maxPoint().y()),
            fmax(box0.maxPoint().z(), box1.maxPoint().z()));

        return { smallPoint, bigPoint };
	}
}
