#pragma once

#include <array>

#define SimdInsteadUnroll 0

namespace SimdHelper
{
	template <typename Sample, size_t Size>
	struct simd_t
	{
	private:
		template <typename RhsType, typename OP>
		static constexpr simd_t BinaryOP(const simd_t& a, const RhsType& b, OP op)
		{
			simd_t res;
#if SimdInsteadUnroll
#pragma omp simd
#else 
#pragma unroll
#endif
			for (size_t i = 0; i < Size; ++i)
			{
				if constexpr (std::is_same_v<std::remove_cvref_t<RhsType>, simd_t>)
					res.dat[i] = op(a.dat[i], b.dat[i]);
				else
					res.dat[i] = op(a.dat[i], b);
			}
			return res;
		}
		template <typename RhsType, typename OP>
		constexpr simd_t& BinaryOP_Assign(const RhsType& b, OP op)
		{
#if SimdInsteadUnroll
#pragma omp simd
#else 
#pragma unroll
#endif
			for (size_t i = 0; i < Size; ++i)
			{
				if constexpr (std::is_same_v<std::remove_cvref_t<RhsType>, simd_t>)
					dat[i] = op(dat[i], b.dat[i]);
				else
					dat[i] = op(dat[i], b);
			}
			return *this;
		}
	public:
		std::array<Sample, Size> dat;
		constexpr simd_t() = default;
		constexpr simd_t(const std::array<Sample, Size>& init_data) : dat(init_data) {}
		template <typename... Ts> requires (sizeof...(Ts) == Size)
			constexpr simd_t(Ts... xs) : dat{ static_cast<Sample>(xs)... } {}
		constexpr explicit simd_t(Sample x) { dat.fill(x); }

		Sample Sum() const
		{
			Sample y = 0;
#if SimdInsteadUnroll
#pragma omp simd
#else 
#pragma unroll
#endif
			for (size_t i = 0; i < Size; ++i) y += dat[i];
			return y;
		}

		template <typename OP>
		constexpr simd_t BatchFunc(OP op)
		{
			simd_t res;
#if SimdInsteadUnroll
#pragma omp simd
#else 
#pragma unroll
#endif
			for (size_t i = 0; i < Size; ++i) res.dat[i] = op(dat[i]);
			return res;
		}

		template <typename RhsType>
		constexpr simd_t operator+(const RhsType& rhs) const
		{
			return BinaryOP(*this, rhs, [](auto a, auto b) constexpr {return a + b; });
		}
		template <typename RhsType>
		constexpr simd_t& operator+=(const RhsType& rhs)
		{
			return BinaryOP_Assign(rhs, [](auto a, auto b) constexpr {return a + b; });
		}

		template <typename RhsType>
		constexpr simd_t operator-(const RhsType& rhs) const
		{
			return BinaryOP(*this, rhs, [](auto a, auto b) constexpr {return a - b; });
		}
		template <typename RhsType>
		constexpr simd_t& operator-=(const RhsType& rhs)
		{
			return BinaryOP_Assign(rhs, [](auto a, auto b) constexpr {return a - b; });
		}

		template <typename RhsType>
		constexpr simd_t operator*(const RhsType& rhs) const
		{
			return BinaryOP(*this, rhs, [](auto a, auto b) constexpr {return a * b; });
		}
		template <typename RhsType>
		constexpr simd_t& operator*=(const RhsType& rhs)
		{
			return BinaryOP_Assign(rhs, [](auto a, auto b) constexpr {return a * b; });
		}

		template <typename RhsType>
		constexpr simd_t operator/(const RhsType& rhs) const
		{
			return BinaryOP(*this, rhs, [](auto a, auto b) constexpr {return a / b; });
		}
		template <typename RhsType>
		constexpr simd_t& operator/=(const RhsType& rhs)
		{
			return BinaryOP_Assign(rhs, [](auto a, auto b) constexpr {return a / b; });
		}

		constexpr simd_t operator+() const
		{
			return *this;
		}
		constexpr simd_t operator-() const
		{
			simd_t res;
#if SimdInsteadUnroll
#pragma omp simd
#else 
#pragma unroll
#endif
			for (size_t i = 0; i < Size; ++i) res.dat[i] = -dat[i];
			return res;
		}
	};
	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  sin(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::sin(x); }); }
	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  cos(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::cos(x); }); }
	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  tan(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::tan(x); }); }

	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  asin(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::asin(x); }); }
	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  acos(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::acos(x); }); }
	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  atan(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::atan(x); }); }

	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  sinh(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::sinh(x); }); }
	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  cosh(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::cosh(x); }); }
	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  tanh(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::tanh(x); }); }

	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  sqrt(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::sqrt(x); }); }

	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  ceil(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::ceil(x); }); }
	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  floor(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::floor(x); }); }
	template <typename Sample, size_t Size>
	simd_t<Sample, Size>  abs(simd_t<Sample, Size>  x) { return x.BatchFunc([](auto x) {return std::abs(x); }); }
}