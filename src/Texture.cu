#include <Texture.cuh>

namespace TinyRT {
	__device__ Color CheckerTexture::value(float u, float v, const Point3& point) const {
		const float sines = sinf(10 * point.x()) * sinf(10 * point.y()) * sinf(10 * point.z());
		return sines < 0 ? _oddTexture->value(u, v, point) : _evenTexture->value(u, v, point);
	}
}
