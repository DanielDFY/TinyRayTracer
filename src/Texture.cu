#include <Texture.cuh>

namespace TinyRT {
	__device__ Color CheckerTexture::value(float u, float v, const Point3& point) const {
		const float sines = sinf(10 * point.x()) * sinf(10 * point.y()) * sinf(10 * point.z());
		return sines < 0 ? _oddTexture->value(u, v, point) : _evenTexture->value(u, v, point);
	}

	__device__ Color ImageTexture::value(float u, float v, const Point3& point) const {
		// If we have no texture data, then return solid cyan as a debugging aid.
		if (_textureObject == NULL)
			return { 0.0, 1.0, 1.0 };

		// Clamp input texture coordinates to [0, 1] x [0, 1]
		u = 1.0 - clamp(u, 0.0, 1.0);	// Flip U to image coordinates
		v = 1.0 - clamp(v, 0.0, 1.0);  // Flip V to image coordinates

		return {
			tex2DLayered<float>(_textureObject, u, v, 0),	//R
			tex2DLayered<float>(_textureObject, u, v, 1),	//G
			tex2DLayered<float>(_textureObject, u, v, 2)		//B
		};
	}

}
