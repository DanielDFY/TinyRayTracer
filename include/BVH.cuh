#pragma once

#include <Hittable.cuh>

namespace TinyRT {
	struct BVHHittableData {
		AABB boundingBox;
		size_t hittableIdx;
	};
	
	struct BVHNodeData {
		bool isLeaf;
		AABB boundingBox;
		// if is leaf, index of hittableIdx
		// if not leaf, index of BVHNodeData
		size_t leftIdx;
		size_t rightIdx;
	};

	inline bool boxCompare(const BVHHittableData& a, const BVHHittableData& b, int axis) {
		return a.boundingBox.minPoint().elem(axis) < b.boundingBox.minPoint().elem(axis);
	}

	inline bool boxCompareX(const BVHHittableData& a, const BVHHittableData& b) {
		return boxCompare(a, b, 0);
	}

	inline bool boxCompareY(const BVHHittableData& a, const BVHHittableData& b) {
		return boxCompare(a, b, 1);
	}

	inline bool boxCompareZ(const BVHHittableData& a, const BVHHittableData& b) {
		return boxCompare(a, b, 2);
	}

	void prepareBVHNodeData(BVHHittableData* bvhHittableDataList, size_t start, size_t end, BVHNodeData* bvhNodeDataList, size_t& currentNodeNum);
	
	size_t prepareBVHNodeData(BVHHittableData* bvhHittableDataList, size_t hittableDataListSize, BVHNodeData* bvhNodeDataList);

	__global__ void buildBVHTree(Hittable** hittablePtrList, BVHNodeData* bvhNodeDataList, size_t nodeNum, Hittable** bvhTree, Hittable** hittableListPtr);
	
	class BVHNode : public Hittable {
	public:
		__device__ BVHNode() : _isLeaf(false), _boundingBox(AABB()), _leftPtr(nullptr), _rightPtr(nullptr) {}
		__device__ BVHNode(bool isLeaf, AABB boundingBox, Hittable* leftPtr, Hittable* rightPtr)
			: _isLeaf(isLeaf), _boundingBox(boundingBox), _leftPtr(leftPtr), _rightPtr(rightPtr) {}

		__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;
		__device__ bool boundingBox(float time0, float time1, AABB& outputBox) const override;

	public:
		bool _isLeaf;
		AABB _boundingBox;
		Hittable* _leftPtr;
		Hittable* _rightPtr;
	};
}
