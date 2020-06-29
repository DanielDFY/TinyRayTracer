#include <Perlin.cuh>

namespace TinyRT {
	__device__ Perlin::Perlin(curandState* randStatePtr) {
		_ranVecList = new Vec3[_pointCount];
		for (int i = 0; i < _pointCount; ++i) {
			_ranVecList[i] = unitVec3(Vec3::random(-1.0f, 1.0f, randStatePtr));
		}

		_permX = perlinGeneratePerm(randStatePtr);
		_permY = perlinGeneratePerm(randStatePtr);
		_permZ = perlinGeneratePerm(randStatePtr);
	}

	__device__ Perlin::~Perlin() {
		delete[] _ranVecList;
		delete[] _permX;
		delete[] _permY;
		delete[] _permZ;
	}

	__device__ float Perlin::noise(const Point3& p) const {
		const float u = p.x() - floor(p.x());
		const float v = p.y() - floor(p.y());
		const float w = p.z() - floor(p.z());

		const auto i = static_cast<int>(floor(p.x()));
		const auto j = static_cast<int>(floor(p.y()));
		const auto k = static_cast<int>(floor(p.z()));

		Vec3 c[2][2][2];

		for (int di = 0; di < 2; ++di)
			for (int dj = 0; dj < 2; ++dj)
				for (int dk = 0; dk < 2; ++dk)
					c[di][dj][dk] = _ranVecList[
						_permX[(i + di) & 255] ^
							_permY[(j + dj) & 255] ^
							_permZ[(k + dk) & 255]
					];

		return perlinInterp(c, u, v, w);
	}

	__device__ float Perlin::turb(const Point3& p, int depth) const {
		float accum = 0.0f;
		Point3 tempPoint = p;
		float weight = 1.0f;

		for (int i = 0; i < depth; ++i) {
			accum += weight * noise(tempPoint);
			weight *= 0.5f;
			tempPoint *= 2;
		}

		return fabs(accum);
	}

	__device__ int* Perlin::perlinGeneratePerm(curandState* randStatePtr) {
		const auto p = new int[_pointCount];

		for (int i = 0; i < _pointCount; ++i) {
			p[i] = i;
		}

		permute(p, _pointCount, randStatePtr);

		return p;
	}

	__device__ void Perlin::permute(int* p, int n, curandState* randStatePtr) {
		for (int i = n - 1; i > 0; --i) {
			int target = randomInt(0, i, randStatePtr);
			int tmp = p[i];
			p[i] = p[target];
			p[target] = tmp;
		}
	}

	__device__ inline float Perlin::perlinInterp(Vec3 c[2][2][2], float u, float v, float w) {
		const float uu = u * u * (3 - 2 * u);
		const float vv = v * v * (3 - 2 * v);
		const float ww = w * w * (3 - 2 * w);
		float accum = 0.0;

		for (int i = 0; i < 2; ++i)
			for (int j = 0; j < 2; ++j)
				for (int k = 0; k < 2; ++k) {
					Vec3 weightVec(u - i, v - j, w - k);
					accum += (i * uu + (1 - i) * (1 - uu))
						* (j * vv + (1 - j) * (1 - vv))
						* (k * ww + (1 - k) * (1 - ww))
						* dot(c[i][j][k], weightVec);
				}


		return accum;
	}
}
