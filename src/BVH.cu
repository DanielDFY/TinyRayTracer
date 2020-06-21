#include <BVH.cuh>

namespace TinyRT {
	void prepareBVHNodeData(BVHHittableData* bvhHittableDataList, size_t start, size_t end, BVHNodeData* bvhNodeDataList, size_t& currentNodeIdx) {
		BVHNodeData& currentNodeData = bvhNodeDataList[currentNodeIdx];
		
		const int axis = randomInt(0, 2);

		const auto comparator = (axis == 0) ? boxCompareX : (axis == 1) ? boxCompareY : boxCompareZ;

		const size_t objectSpan = end - start;

		if (objectSpan == 1) {
			// The traversal algorithm should be smooth and not have to check for null pointers,
			// so if there is only one element, duplicate it in each subtree.
			currentNodeData.isLeaf = true;
			currentNodeData.boundingBox = bvhHittableDataList[start].boundingBox;
			currentNodeData.leftIdx = currentNodeData.rightIdx = bvhHittableDataList[start].hittableIdx;
		}
		else if (objectSpan == 2) {
			currentNodeData.isLeaf = true;
			// There are only two elements
			const size_t second = start + 1;
			if (comparator(bvhHittableDataList[start], bvhHittableDataList[second])) {
				currentNodeData.leftIdx = bvhHittableDataList[start].hittableIdx;
				currentNodeData.rightIdx = bvhHittableDataList[second].hittableIdx;
			}
			else {
				currentNodeData.leftIdx = bvhHittableDataList[second].hittableIdx;
				currentNodeData.rightIdx = bvhHittableDataList[start].hittableIdx;
			}
			const AABB boxLeft = bvhHittableDataList[currentNodeData.leftIdx].boundingBox;
			const AABB boxRight = bvhHittableDataList[currentNodeData.rightIdx].boundingBox;
			currentNodeData.boundingBox = surroundingBox(boxLeft, boxRight);
		} else {
			currentNodeData.isLeaf = false;
			std::sort(bvhHittableDataList + start, bvhHittableDataList + end, comparator);

			const size_t mid = start + objectSpan / 2;
			currentNodeData.leftIdx = ++currentNodeIdx;
			prepareBVHNodeData(bvhHittableDataList, start, mid, bvhNodeDataList, currentNodeIdx);
			currentNodeData.rightIdx = ++currentNodeIdx;
			prepareBVHNodeData(bvhHittableDataList, mid, end, bvhNodeDataList, currentNodeIdx);

			const AABB boxLeft = bvhNodeDataList[currentNodeData.leftIdx].boundingBox;
			const AABB boxRight = bvhNodeDataList[currentNodeData.rightIdx].boundingBox;
			currentNodeData.boundingBox = surroundingBox(boxLeft, boxRight);
		}
	}

	size_t prepareBVHNodeData(BVHHittableData* bvhHittableDataList, size_t hittableDataListSize, BVHNodeData* bvhNodeDataList) {
		size_t currentNodeIdx = 0;
		prepareBVHNodeData(bvhHittableDataList, 0, hittableDataListSize, bvhNodeDataList, currentNodeIdx);
		return currentNodeIdx + 1;
	}

	__device__ bool BVHNode::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const {
		if (!_boundingBox.hit(r, tMin, tMax))
			return false;

		BVHNode* stack[64];
		BVHNode** stackPtr = stack;
		*stackPtr++ = nullptr;

		auto nodePtr = this;
		do {
			if (nodePtr->_isLeaf) {
				const bool isLeftHit = nodePtr->_leftPtr->hit(r, tMin, tMax, rec);
				const bool isRightHit = nodePtr->_rightPtr->hit(r, tMin, tMax, rec);

				if (isLeftHit || isRightHit) {
					return true;
				} else {
					nodePtr = *--stackPtr;	// pop
				}
			} else {
				BVHNode* nodeLeftPtr = static_cast<BVHNode*>(nodePtr->_leftPtr);
				BVHNode* nodeRightPtr = static_cast<BVHNode*>(nodePtr->_rightPtr);

				const bool isLeftHit = nodeLeftPtr->_boundingBox.hit(r, tMin, tMax);
				const bool isRightHit = nodeRightPtr->_boundingBox.hit(r, tMin, tMax);

				if (!isLeftHit && !isRightHit) {
					nodePtr = *--stackPtr;	// pop
				}
				else {
					nodePtr = static_cast<BVHNode*>(isLeftHit ? nodePtr->_leftPtr : nodePtr->_rightPtr);
					if (isLeftHit && isRightHit) {
						*stackPtr++ = static_cast<BVHNode*>(nodePtr->_rightPtr);		// push
					}
				}
			}
		} while (nodePtr != nullptr);

		return false;
	}

	__device__ bool BVHNode::boundingBox(float time0, float time1, AABB& outputBox) const {
		outputBox = _boundingBox;
		return true;
	}


}
