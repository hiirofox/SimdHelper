#include "../SimdType.h"
#include <tuple>

namespace NLModeling3
{
	constexpr static int NumLayers = 8;
	constexpr static int FiltOrder = 8;
	constexpr static int ParamsPerLayer = FiltOrder * 2 + 1 + 5 + 3;
	constexpr static int NumParams = NumLayers * ParamsPerLayer;
	constexpr static float kScale = 0.9995;

	struct NLModelParams
	{
		float k[NumLayers][FiltOrder];
		float gf[NumLayers][FiltOrder];
		float gfx[NumLayers];

		float a1[NumLayers];
		float a2[NumLayers];
		float a3[NumLayers];
		float b1[NumLayers];
		float b2[NumLayers];

		float gdry[NumLayers];
		float gnlin[NumLayers];
		float gnlout[NumLayers];

		template<typename Sample>
		static inline Sample Clip(Sample x, Sample lo, Sample hi)
		{
			return x < lo ? lo : (x > hi ? hi : x);
		}
		static void InitVecDirect(float* out)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = 0.0f;

				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = 0.0f;

				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 1.0f;
			}
		}

		void ParamsToVec(float* out) const
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = k[layer][i];

				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = gf[layer][i];

				out[p++] = gfx[layer];
				out[p++] = a1[layer];
				out[p++] = a2[layer];
				out[p++] = a3[layer];
				out[p++] = b1[layer];
				out[p++] = b2[layer];
				out[p++] = gdry[layer];
				out[p++] = gnlin[layer];
				out[p++] = gnlout[layer];
			}
		}

		void VecToParams(const float* in)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int i = 0; i < FiltOrder; ++i)
					k[layer][i] = Clip(in[p++], -kScale, kScale);

				for (int i = 0; i < FiltOrder; ++i)
					gf[layer][i] = in[p++];

				gfx[layer] = in[p++];
				a1[layer] = in[p++];
				a2[layer] = in[p++];
				a3[layer] = in[p++];
				b1[layer] = in[p++];
				b2[layer] = in[p++];
				gdry[layer] = in[p++];
				gnlin[layer] = in[p++];
				gnlout[layer] = in[p++];
			}
		}
	};

	template<int layer, typename Sample>
	inline std::tuple<Sample, Sample> ProcessLattice(
		Sample x, Sample* z, const Sample* k,
		const Sample* gf, Sample gfx)
	{
		if constexpr (layer >= FiltOrder)
			return { x, x * gfx };
		else
		{
			auto z0 = z[layer];
			auto a = z0 * k[layer] + x;
			auto [nextz, out] = ProcessLattice<layer + 1, Sample>(a, z, k, gf, gfx);
			z[layer] = nextz;
			auto y = a * -k[layer] + z0;
			return { y, out + y * gf[layer] };
		}
	}

	class NLModelProcess
	{
	private:
		float z[NumLayers][FiltOrder];

	public:
		NLModelProcess()
		{
			Init();
		}

		void Init()
		{
			for (int layer = 0; layer < NumLayers; ++layer)
				for (int i = 0; i < FiltOrder; ++i)
					z[layer][i] = 0.0f;
		}

		static inline float Nonlinear(
			float x, const NLModelParams& p, int layer)
		{
			auto x2 = x * x;
			auto absx = std::abs(x);
			auto num = x * (x * (x + p.a1[layer]) + p.a2[layer] + p.a3[layer] * absx);
			auto den = p.b1[layer] * x2 + p.b2[layer] * x2 * absx + 1.0f;
			return num / den;
		}

		inline float ProcessCell(float x, const NLModelParams& p, int layer)
		{
			auto [nextz, latticeOut] = ProcessLattice<0, float>(
				x, z[layer], p.k[layer], p.gf[layer], p.gfx[layer]);
			auto nlo = Nonlinear(latticeOut * p.gnlin[layer], p, layer);
			return x * p.gdry[layer] + nlo * p.gnlout[layer];
		}

		void ProcessBlock(NLModelParams& p, const float* in, float* out, int NumSamples)
		{
			for (int i = 0; i < NumSamples; ++i)
			{
				float x = in[i];
				for (int layer = 0; layer < NumLayers; ++layer)
					x = ProcessCell(x, p, layer);
				out[i] = x;
			}
		}

		constexpr static int GetTargetDelaySample()//ÑµÁ·×î¼Ñ¶ÔÆë
		{
			return 1;
		}
		constexpr static int GetDelaySample()//ÑÓ³Ù
		{
			return 0;
		}
	};
}


namespace NLModeling3Fast
{
	constexpr static int NumLayers = 8;
	constexpr static int FiltOrder = 8;
	constexpr static int ParamsPerLayer = FiltOrder * 2 + 1 + 5 + 3;
	constexpr static int NumParams = NumLayers * ParamsPerLayer;
	constexpr static float kScale = 0.9995;

	struct NLModelParams
	{
		float k[NumLayers][FiltOrder];
		float gf[NumLayers][FiltOrder];
		float gfx[NumLayers];

		float a1[NumLayers];
		float a2[NumLayers];
		float a3[NumLayers];
		float b1[NumLayers];
		float b2[NumLayers];

		float gdry[NumLayers];
		float gnlin[NumLayers];
		float gnlout[NumLayers];

		template<typename Sample>
		static inline Sample Clip(Sample x, Sample lo, Sample hi)
		{
			return x < lo ? lo : (x > hi ? hi : x);
		}
		static void InitVecDirect(float* out)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = 0.0f;

				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = 0.0f;

				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 0.0f;
				out[p++] = 0.0f;
				out[p++] = 1.0f;
				out[p++] = 1.0f;
			}
		}

		void ParamsToVec(float* out) const
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = k[layer][i];

				for (int i = 0; i < FiltOrder; ++i)
					out[p++] = gf[layer][i];

				out[p++] = gfx[layer];
				out[p++] = a1[layer];
				out[p++] = a2[layer];
				out[p++] = a3[layer];
				out[p++] = b1[layer];
				out[p++] = b2[layer];
				out[p++] = gdry[layer];
				out[p++] = gnlin[layer];
				out[p++] = gnlout[layer];
			}
		}

		void VecToParams(const float* in)
		{
			int p = 0;
			for (int layer = 0; layer < NumLayers; ++layer)
			{
				for (int i = 0; i < FiltOrder; ++i)
					k[layer][i] = Clip(in[p++], -kScale, kScale);

				for (int i = 0; i < FiltOrder; ++i)
					gf[layer][i] = in[p++];

				gfx[layer] = in[p++];
				a1[layer] = in[p++];
				a2[layer] = in[p++];
				a3[layer] = in[p++];
				b1[layer] = in[p++];
				b2[layer] = in[p++];
				gdry[layer] = in[p++];
				gnlin[layer] = in[p++];
				gnlout[layer] = in[p++];
			}
		}
	};


	class NLModelProcess
	{
	private:
		using Sample = SimdHelper::simd_t<float, NumLayers>;
		Sample a1, a2, a3, b1, b2;
		Sample k[FiltOrder], gf[FiltOrder], gfx, gdry, gnlin, gnlout;
		Sample z[FiltOrder];
		Sample x;
	public:
		NLModelProcess()
		{
			Init();
		}

		void Init()
		{
			for (auto& v : z)v.fill(0.0f);
			x.fill(0.0f);
		}
		template<int layer>
		inline std::tuple<Sample, Sample> ProcessLattice(
			Sample x, Sample* z, const Sample* k,
			const Sample* gf, Sample gfx)
		{
			if constexpr (layer >= FiltOrder)
				return { x, x * gfx };
			else
			{
				auto z0 = z[layer];
				auto a = z0 * k[layer] + x;
				auto [nextz, out] = ProcessLattice<layer + 1>(a, z, k, gf, gfx);
				z[layer] = nextz;
				auto y = a * -k[layer] + z0;
				return { y, out + y * gf[layer] };
			}
		}
		inline Sample Nonlinear(Sample x)
		{
			Sample x2 = x * x;
			Sample absx = abs(x);
			Sample num = x * (x * (x + a1) + a2 + a3 * absx);
			Sample den = b1 * x2 + b2 * x2 * absx + 1.0f;
			return num / den;
		}
		inline Sample ProcessCell(Sample x)
		{
			auto [nextz, latticeOut] = ProcessLattice<0>(x, z, k, gf, gfx);
			auto nlo = Nonlinear(latticeOut * gnlin);
			return x * gdry + nlo * gnlout;
		}
		void UpdateParams(NLModelParams& p)
		{
			for (int i = 0; i < NumLayers; ++i)
			{
				a1.dat[i] = p.a1[i];
				a2.dat[i] = p.a2[i];
				a3.dat[i] = p.a3[i];
				b1.dat[i] = p.b1[i];
				b2.dat[i] = p.b2[i];
				gfx.dat[i] = p.gfx[i];
				gdry.dat[i] = p.gdry[i];
				gnlin.dat[i] = p.gnlin[i];
				gnlout.dat[i] = p.gnlout[i];
			}
			for (int i = 0; i < NumLayers; ++i)
			{
				for (int j = 0; j < FiltOrder; ++j)
				{
					k[j].dat[i] = p.k[i][j];
					gf[j].dat[i] = p.gf[i][j];
				}
			}
		}
		void ProcessBlock(NLModelParams& p, const float* in, float* out, int NumSamples)
		{
			UpdateParams(p);
			for (int i = 0; i < NumSamples; ++i)
			{
				x.dat[0] = in[i];
				x = ProcessCell(x);
				out[i] = x.dat[NumLayers - 1];
				x.ShiftRight();
			}
		}

		constexpr static int GetTargetDelaySample()//ÑµÁ·×î¼Ñ¶ÔÆë
		{
			return 1 + NumLayers;
		}
		constexpr static int GetDelaySample()//ÑÓ³Ù
		{
			return NumLayers - 1;
		}
	};
}
constexpr static int NumParams = 200;
constexpr static float nl3_nd_d10_t10[NumParams] =
{
	0.22143845f,0.28913817f,0.37612954f,0.39290699f,0.34039164f,0.26385364f,0.14040197f,-0.26186770f,
	-0.07422879f,-0.43524399f,-0.43706188f,-0.43909591f,-0.40666181f,-0.32358220f,-0.28806433f,-0.11937921f,
	1.35893476f,0.01535577f,1.21793962f,0.23746969f,0.85054624f,0.06092754f,0.06980757f,1.22884786f,
	1.21528971f,0.00682158f,0.29452410f,0.19970743f,0.07497364f,-0.14688782f,0.10878590f,0.23308440f,
	0.00939015f,-0.26497233f,-0.65849376f,-0.40549776f,-0.34732613f,-0.00346886f,-0.57322156f,-0.48809293f,
	0.02506791f,1.54119599f,-0.00659241f,1.27133501f,-0.40910625f,1.14362359f,-0.01053994f,0.18369576f,
	1.33093417f,1.20156693f,-0.01819783f,0.26626951f,0.36380604f,-0.28699642f,0.05374550f,0.18034011f,
	0.24964423f,0.15459163f,-0.13490975f,-0.81600487f,-0.54480165f,0.38310361f,-0.17275643f,-0.24869055f,
	-0.49799594f,0.31739098f,1.31307924f,-0.03913219f,1.16136706f,-0.75263733f,1.41579950f,0.52095801f,
	0.53412378f,1.27416360f,0.92038858f,0.03717420f,-0.00115456f,0.40129074f,0.15487298f,0.22559908f,
	0.23513103f,-0.09678576f,0.36421260f,-0.04021091f,-0.16531074f,-0.77586657f,0.08032236f,-0.72214234f,
	0.33016387f,-0.57745332f,0.38977697f,0.89672047f,0.01240012f,1.42750478f,-0.33851060f,0.96445054f,
	0.07235660f,0.08594362f,1.30711555f,1.33257723f,0.07562763f,-0.25982669f,0.42133763f,0.14076975f,
	0.08264019f,0.34176806f,-0.12291563f,0.41446969f,-0.22132559f,-0.12565370f,-0.54025692f,-0.45617241f,
	-0.11222795f,-0.35230142f,0.35883993f,0.41883057f,0.55719894f,0.01679084f,1.41902637f,0.30787712f,
	0.94034612f,0.91295177f,0.64209694f,1.07806301f,0.22575833f,0.03841212f,0.14305972f,0.35674325f,
	0.06649885f,0.26074365f,0.06555130f,0.15309615f,0.12143426f,-0.13102542f,-0.36947230f,-0.44039661f,
	-0.16975754f,-0.29387397f,-0.12697247f,-0.37139234f,0.44611439f,0.79228210f,0.00150333f,1.09402287f,
	0.04185031f,1.02148974f,-0.06205936f,-0.08649572f,1.10175514f,1.07355690f,0.03904330f,0.28800878f,
	0.27037272f,0.11359705f,0.57009459f,-0.02313068f,0.38823783f,0.00474571f,-0.01499280f,-0.11087144f,
	-0.11364785f,-0.02310645f,-0.20949371f,0.01561504f,-0.08983350f,-0.00090293f,0.20099820f,-0.00053319f,
	1.04630113f,0.03174156f,1.10824966f,0.00416697f,-0.31004113f,0.95128775f,0.95862806f,-0.08877098f,
	-0.10026165f,0.03281452f,0.34769806f,0.14306881f,0.31397909f,0.38753241f,0.24977490f,-0.12798676f,
	-0.22771209f,-0.26234126f,-0.15558290f,-0.24953327f,0.05759279f,0.23226760f,0.10817992f,0.16915320f,
	-0.01367395f,0.96056288f,0.04756189f,0.91726226f,-0.04540731f,-0.07084839f,1.03305662f,1.02283394f
};