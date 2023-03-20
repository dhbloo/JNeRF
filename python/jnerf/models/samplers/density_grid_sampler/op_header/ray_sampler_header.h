#include <atomic>
#include <limits>
#include <stdexcept>
#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "utils/log.h"
#include "cuda_fp16.h"
#include "pcg32.h"
#include "nerf_common.h"

using namespace Eigen;

#define TCNN_HOST_DEVICE __host__ __device__
#define TCNN_MIN_GPU_ARCH 70

inline __device__ int mip_from_pos(const Vector3f &pos)
{
	int exponent;
	float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
	frexpf(maxval, &exponent);
	return min(NERF_CASCADES() - 1, max(0, exponent + 1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f &pos)
{
	int mip = mip_from_pos(pos);
	dt *= 2 * NERF_GRIDSIZE();
	if (dt < 1.f)
		return mip;
	int exponent;
	frexpf(dt, &exponent);
	return min(NERF_CASCADES() - 1, max(exponent, mip));
}

// ==========================
// other needed structure

struct CameraDistortion
{
	float params[4] = {};
#ifdef __NVCC__
	inline __host__ __device__ bool is_zero() const
	{
		return params[0] == 0.0f && params[1] == 0.0f && params[2] == 0.0f && params[3] == 0.0f;
	}
#endif
};

struct TrainingImageMetadata
{
	// Camera intrinsics and additional data associated with a NeRF training image
	CameraDistortion camera_distortion = {};
	Eigen::Vector2f principal_point = Eigen::Vector2f::Constant(0.5f);
	Eigen::Vector2f focal_length = Eigen::Vector2f::Constant(1000.f);

	// TODO: replace this with more generic float[] of task-specific metadata.
	Eigen::Vector3f light_dir = Eigen::Vector3f::Constant(0.f);
};

template <typename T>
struct PitchedPtr
{
	TCNN_HOST_DEVICE PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
	TCNN_HOST_DEVICE PitchedPtr(T *ptr, size_t stride_in_elements, size_t offset = 0, size_t extra_stride_bytes = 0) : ptr{ptr + offset}, stride_in_bytes{(uint32_t)(stride_in_elements * sizeof(T) + extra_stride_bytes)} {}

	template <typename U>
	TCNN_HOST_DEVICE explicit PitchedPtr(PitchedPtr<U> other) : ptr{(T *)other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

	TCNN_HOST_DEVICE T *operator()(uint32_t y) const
	{
		return (T *)((const char *)ptr + y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE void operator+=(uint32_t y)
	{
		ptr = (T *)((const char *)ptr + y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE void operator-=(uint32_t y)
	{
		ptr = (T *)((const char *)ptr - y * stride_in_bytes);
	}

	TCNN_HOST_DEVICE explicit operator bool() const
	{
		return ptr;
	}

	T *ptr;
	uint32_t stride_in_bytes;
};

template <typename T, uint32_t N_ELEMS>
struct vector_t
{
	TCNN_HOST_DEVICE T &operator[](uint32_t idx)
	{
		return data[idx];
	}

	TCNN_HOST_DEVICE T operator[](uint32_t idx) const
	{
		return data[idx];
	}

	T data[N_ELEMS];
	static constexpr uint32_t N = N_ELEMS;
};

// ==========================
// other needed functions

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ __host__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 8; }

// Any alpha below this is considered "invisible" and is thus culled away.
// inline constexpr __device__ __host__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

inline __host__ __device__ float calc_cone_angle(float cosine, const Eigen::Vector2f &focal_length, float cone_angle_constant)
{
	// Pixel size. Doesn't always yield a good performance vs. quality
	// trade off. Especially if training pixels have a much different
	// size than rendering pixels.
	// return cosine*cosine / focal_length.mean();

	return cone_angle_constant;
}

inline __device__ float distance_to_next_voxel(const Vector3f &pos, const Vector3f &dir, const Vector3f &idir, uint32_t res)
{ // dda like step
	Vector3f p = res * pos;
	float tx = (floorf(p.x() + 0.5f + 0.5f * sign(dir.x())) - p.x()) * idir.x();
	float ty = (floorf(p.y() + 0.5f + 0.5f * sign(dir.y())) - p.y()) * idir.y();
	float tz = (floorf(p.z() + 0.5f + 0.5f * sign(dir.z())) - p.z()) * idir.z();
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(float t, float cone_angle, const Vector3f &pos, const Vector3f &dir, const Vector3f &idir, uint32_t res)
{
	// Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
	// due to the different stepping.
	// float dt = calc_dt(t, cone_angle);
	// return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

	// Regular stepping (may be slower but matches non-empty space)
	float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
	do
	{
		t += calc_dt(t, cone_angle);
	} while (t < t_target);
	return t;
}

inline __device__ uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip)
{
	float mip_scale = scalbnf(1.0f, -mip);
	pos -= Vector3f::Constant(0.5f);
	pos *= mip_scale;
	pos += Vector3f::Constant(0.5f);

	Vector3i i = (pos * NERF_GRIDSIZE()).cast<int>();

	uint32_t idx = morton3D(
		clamp(i.x(), 0, (int)NERF_GRIDSIZE() - 1),
		clamp(i.y(), 0, (int)NERF_GRIDSIZE() - 1),
		clamp(i.z(), 0, (int)NERF_GRIDSIZE() - 1));

	return idx;
}

inline __device__ bool density_grid_occupied_at(const Vector3f &pos, const uint8_t *density_grid_bitfield, uint32_t mip)
{
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return density_grid_bitfield[idx / 8 + grid_mip_offset(mip) / 8] & (1 << (idx % 8));
}

inline __device__ float cascaded_grid_at(Vector3f pos, const float *cascaded_grid, uint32_t mip)
{
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx + grid_mip_offset(mip)];
}

inline __device__ float &cascaded_grid_at(Vector3f pos, float *cascaded_grid, uint32_t mip)
{
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx + grid_mip_offset(mip)];
}

// ==========================
// network activation

inline __device__ float network_to_density(float val, ENerfActivation activation)
{
	return apply_activation(val, activation);
}

inline __device__ float network_to_rgb(float val, ENerfActivation activation)
{
	switch (activation)
	{
	case ENerfActivation::Exponential:
		return apply_activation(clamp(val, -10.0f, 10.0f), activation);
	default:
		return apply_activation(val, activation);
	}
}

template <typename T>
inline __device__ Array3f network_to_rgb(const vector_t<T, 4> &local_network_output, ENerfActivation activation)
{
	return {
		network_to_rgb(float(local_network_output[0]), activation),
		network_to_rgb(float(local_network_output[1]), activation),
		network_to_rgb(float(local_network_output[2]), activation)
	};
}

inline __device__ float network_to_rgb_derivative(float val, ENerfActivation activation)
{
	switch (activation)
	{
	case ENerfActivation::Exponential:
		return activation_derivative(clamp(val, -10.0f, 10.0f), activation);
	default:
		return activation_derivative(val, activation);
	}
}

inline __device__ float network_to_density_derivative(float val, ENerfActivation activation)
{
	switch (activation)
	{
	case ENerfActivation::Exponential:
		return activation_derivative(clamp(val, -15.0f, 15.0f), activation);
	default:
		return activation_derivative(val, activation);
	}
}
