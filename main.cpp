#include "SimdType.h"
#include <tuple>
#include <chrono>

#include "test/NLModeling.h"
#include "test/GranularPitchShifter.h"

constexpr static int BenchSize = 48000 * 1000;
float test[BenchSize];
float output1[BenchSize];
float output2[BenchSize];

namespace NLModeling3Bench
{
	NLModeling3::NLModelParams nl3Params;
	NLModeling3Fast::NLModelParams nl3FastParams;
	NLModeling3::NLModelProcess nl3Proc;
	NLModeling3Fast::NLModelProcess nl3FastProc;
	void Bench()
	{
		nl3Params.VecToParams(nl3_nd_d10_t10);
		nl3FastParams.VecToParams(nl3_nd_d10_t10);
		nl3Proc.Init();
		nl3FastProc.Init();

		for (int i = 0; i < BenchSize; ++i) test[i] = (float)(rand() % 10000) / 50000.0 * (rand() % 2 ? 1 : -1);

		auto start1 = std::chrono::steady_clock::now();
		nl3Proc.ProcessBlock(nl3Params, test, output1, BenchSize);
		auto end1 = std::chrono::steady_clock::now();
		auto start2 = std::chrono::steady_clock::now();
		nl3FastProc.ProcessBlock(nl3FastParams, test, output2, BenchSize);
		auto end2 = std::chrono::steady_clock::now();
		long long elap1 = (end1 - start1).count();
		long long elap2 = (end2 - start2).count();
		int benchEnd = BenchSize - (std::max)(nl3Proc.GetDelaySample(), nl3FastProc.GetDelaySample());
		double mse1 = 0, mse2 = 0, mse3 = 0;
		for (int i = 0; i < benchEnd; ++i)
		{
			int idx1 = i + nl3Proc.GetDelaySample();
			int idx2 = i + nl3FastProc.GetDelaySample();
			double y1 = output1[idx1];
			double y2 = output2[idx2];
			mse1 += y1 * y1;
			mse2 += y2 * y2;
			mse3 += (y1 - y2) * (y1 - y2);
		}
		mse1 /= benchEnd;
		mse2 /= benchEnd;
		mse3 /= benchEnd;
		printf("NLModeling3:\n");
		printf("Origin clock:%lld; SimdHelper clock:%lld (speedup:%.5fX)\n", elap1, elap2, (double)elap1 / elap2);
		printf("Origin mse:%e; SimdHelper mes:%e; difference mse:%e\n", mse1, mse2, mse3);
	}
}
namespace GranularBench
{
	GranularPitchShifterParams gpsParams;
	GranularPitchShifter<float> gpsl, gpsr;
	using float2ch = SimdHelper::simd_t<float, 2>;
	GranularPitchShifter<float2ch> gps2ch;
	float2ch test2ch[BenchSize];
	float2ch output2ch[BenchSize];
	void Bench()
	{
		gpsl.Init();
		gpsr.Init();
		gps2ch.Init();
		for (int i = 0; i < BenchSize; ++i)
		{
			test[i] = (float)(rand() % 10000) / 50000.0 * (rand() % 2 ? 1 : -1);
			test2ch[i] = test[i];
		}

		auto start1 = std::chrono::steady_clock::now();
		srand(12345);
		gpsl.ProcessBlock(gpsParams, test, output1, BenchSize);
		srand(12345);
		gpsr.ProcessBlock(gpsParams, test, output2, BenchSize);
		for (int i = 0; i < BenchSize; ++i) output1[i] += output2[i];
		auto end1 = std::chrono::steady_clock::now();
		auto start2 = std::chrono::steady_clock::now();
		srand(12345);
		srand(12345);
		gps2ch.ProcessBlock(gpsParams, test2ch, output2ch, BenchSize);
		for (int i = 0; i < BenchSize; ++i) output2[i] = output2ch[i].dat[0] + output2ch[i].dat[1];
		auto end2 = std::chrono::steady_clock::now();

		long long elap1 = (end1 - start1).count();
		long long elap2 = (end2 - start2).count();
		int benchEnd = BenchSize;
		double mse1 = 0, mse2 = 0, mse3 = 0;
		for (int i = 0; i < benchEnd; ++i)
		{
			double y1 = output1[i];
			double y2 = output2[i];
			mse1 += y1 * y1;
			mse2 += y2 * y2;
			mse3 += (y1 - y2) * (y1 - y2);
		}
		mse1 /= benchEnd;
		mse2 /= benchEnd;
		mse3 /= benchEnd;
		printf("GranularPitchShift:\n");
		printf("Origin clock:%lld; SimdHelper clock:%lld (speedup:%.5fX)\n", elap1, elap2, (double)elap1 / elap2);
		printf("Origin mse:%e; SimdHelper mes:%e; difference mse:%e\n", mse1, mse2, mse3);
	}
}
int main()
{
	//NLModeling3Bench::Bench();
	GranularBench::Bench();
}