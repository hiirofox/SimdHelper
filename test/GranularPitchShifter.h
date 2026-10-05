#pragma once

#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <array>

#pragma once

template<typename Sample>
class HalfNyquistLowpass
{
private:
	constexpr static Sample B01 = (Sample)0.2598915324741450;
	constexpr static Sample A21 = (Sample)0.0395661298965800;
	constexpr static Sample B02 = (Sample)0.3616156730429224;
	constexpr static Sample A22 = (Sample)0.4464626921716895;
	Sample x11 = 0, x12 = 0;
	Sample x21 = 0, x22 = 0;
	Sample y12 = 0;
	Sample y22 = 0;
public:
	HalfNyquistLowpass()
	{
		Reset();
	}
	void Reset()
	{
		x11 = x12 = 0;
		y12 = 0;
		x21 = x22 = 0;
		y22 = 0;
	}
	Sample ProcessSample(Sample x)
	{
		Sample y1 = (x + x11 + x11 + x12) * B01 - y12 * A21;
		Sample y2 = (y1 + x21 + x21 + x22) * B02 - y22 * A22;
		x12 = x11;
		x11 = x;
		y12 = y1;
		x22 = x21;
		x21 = y1;
		y22 = y2;
		return y2;
	}
};

template<typename Sample>
class DeDCHighpass
{
private:
	Sample z = 0;
public:
	DeDCHighpass()
	{
		Reset();
	}
	void Reset()
	{
		z = 0;
	}
	Sample ProcessSample(Sample x)
	{
		z += (x - z) * 0.001;
		return x - z;
	}
};

struct GranularPitchShifterParams
{
	float pitch = 2.0;//八度
	float pitchRandX = 0.1;//音高随机化(0~1)
	float grainSizeX = 1.0;//粒子大小(0~1)
	float grainSamplingRangeX = 1.0;//粒子采集区间(0~1)
};

template<typename Sample>
class GranularPitchShifter
{
private:
	constexpr static int MaxBufferSize = 32768;
	constexpr static int MaxGrainSize = 8192;
	constexpr static int NumGrains = 2;
	//window data
	std::array<Sample, MaxGrainSize> window;
	void InitWindow(int grainSize)
	{
		for (int i = 0; i < grainSize; ++i)
		{
			Sample x = (Sample)i / grainSize * 2.0 - 1.0;
			x = -x * x + 1.0;
			x = x * x;
			window[i] = x;//square welch
		}
	}
	Sample ReadWindow(float t)
	{
		int idx = t;
		Sample frac = t - idx;
		int idx1 = (idx + 0) % grainSize;
		int idx2 = (idx + 1) % grainSize;
		Sample y1 = window[idx1];
		Sample y2 = window[idx2];
		return y1 + (y2 - y1) * frac;
	}
	//buffer data
	std::array<Sample, MaxBufferSize> buffer;
	int writePos = 0;
	void WriteBuffer(Sample x)
	{
		buffer[writePos] = x;
		writePos++;
		if (writePos >= MaxBufferSize)writePos = 0;
	}
	Sample ReadBuffer(float t)
	{
		int idx = t;
		Sample frac = t - idx;
		int idx1 = (MaxBufferSize + writePos + idx + 0) % MaxBufferSize;
		int idx2 = (MaxBufferSize + writePos + idx + 1) % MaxBufferSize;
		Sample y1 = buffer[idx1];
		Sample y2 = buffer[idx2];
		return y1 + (y2 - y1) * frac;
	}
	//grain data
	float grainStart[NumGrains];
	int grainPos[NumGrains];
	float grainPosMul[NumGrains];

	int lastGrainSize = 0, grainSize = 0;
	float pitch = 2.0, pitchRandX = 0.1;
	float sampRange = 1.0;
	void UpdateGrain(int n)
	{
		grainPos[n] = 0;
		/*
		for (int j = 0; j < NumGrains; ++j)
		{
			int idx = (NumGrains + n - j) % NumGrains;
			grainPos[j] = idx * grainSize / NumGrains;
		}
		*/
		float randv = (float)rand() / RAND_MAX * 0.1 * (rand() % 2 ? 1 : -1);
		float pitchVal = pitch + randv * pitchRandX - 1.0;
		grainPosMul[n] = pitchVal;

		randv = (float)rand() / RAND_MAX;
		float right = MaxBufferSize - grainSize * pitchVal - 10;//10是余量
		float left = right * (1.0 - sampRange);
		grainStart[n] = randv * (right - left) + left;
	}

	HalfNyquistLowpass<Sample> lpf;
	DeDCHighpass<Sample> hpf;
	Sample NormValueMul = 1.0f;
public:
	GranularPitchShifter()
	{
		Init();
	}
	void CheckGrainSizeChange()
	{
		if (grainSize != lastGrainSize)
		{
			lastGrainSize = grainSize;
			InitWindow(grainSize);
			for (int j = 0; j < NumGrains; ++j)
			{
				UpdateGrain(j);
				grainPos[j] = (float)j * grainSize / NumGrains;
			}
		}
	}
	void Init()
	{
		for (auto& v : buffer)v = 0;
		for (auto& v : grainStart)v = 0;
		for (auto& v : grainPosMul)v = 0;
		lpf.Reset();
	}
	void ProcessBlock(GranularPitchShifterParams p, const Sample* in, Sample* out, int numSamples)
	{
		ApplyParams(p);
		for (int i = 0; i < numSamples; ++i)
		{
			out[i] = ProcessSample(in[i]);
		}
	}
	void ApplyParams(GranularPitchShifterParams p)
	{
		grainSize = p.grainSizeX * MaxGrainSize;
		pitch = p.pitch;
		pitchRandX = p.pitchRandX;
		sampRange = p.grainSamplingRangeX;
		NormValueMul = 1.0 / pitch / NumGrains * 2.0;
		CheckGrainSizeChange();
	}
	Sample ProcessSample(Sample x)
	{
		WriteBuffer(lpf.ProcessSample(hpf.ProcessSample(x)));
		Sample y = 0;
		for (int j = 0; j < NumGrains; ++j)
		{
			grainPos[j]++;
			if (grainPos[j] >= grainSize) UpdateGrain(j);
			//Sample w = ReadWindow(grainPos[j]);
			Sample w = window[grainPos[j]];
			Sample v = ReadBuffer(grainStart[j] + (float)grainPos[j] * grainPosMul[j]);
			y += w * v;
		}
		return y * NormValueMul;
	}
};