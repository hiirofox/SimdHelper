// Self-contained evaluation for SimdHelper::simd_t.
// clang: clang++ simdHelperEval.cpp -std=c++20 -O3 -march=native -fopenmp-simd -DNDEBUG -o simdHelperEval
// Note: Clang 17 rejects #pragma omp simd inside constexpr functions, so the runtime operators below
// intentionally omit constexpr. This does not change runtime semantics; MSVC users can restore it if desired.
// MSVC : cl /std:c++20 /O2 /arch:AVX2 /DNDEBUG /EHsc simdHelperEval.cpp

#include <array>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>
#include "SimdType.h"

#ifndef SIMD_EVAL_NUM_PROCS
#define SIMD_EVAL_NUM_PROCS 37
#endif
#ifndef SIMD_EVAL_BLOCK
#define SIMD_EVAL_BLOCK 4096
#endif
#ifndef SIMD_EVAL_PASSES
#define SIMD_EVAL_PASSES 1400
#endif

static constexpr size_t NumProcs = SIMD_EVAL_NUM_PROCS;
static constexpr size_t BlockSize = SIMD_EVAL_BLOCK;
static constexpr int Passes = SIMD_EVAL_PASSES;
static_assert(NumProcs >= 2);

using SimdF = SimdHelper::simd_t<float, NumProcs>;

template<class T> static inline T splat(float x) { return T(x); }
template<> inline float splat<float>(float x) { return x; }

template<class T> static inline T math_sin(T x) { using std::sin;  using SimdHelper::sin;  return sin(x); }
template<class T> static inline T math_cos(T x) { using std::cos;  using SimdHelper::cos;  return cos(x); }
template<class T> static inline T math_tanh(T x) { using std::tanh; using SimdHelper::tanh; return tanh(x); }
template<class T> static inline T math_abs(T x) { using std::abs;  using SimdHelper::abs;  return abs(x); }

// One deliberately busy, streaming DSP processor. It is not intended to be a good audio effect;
// it is intended to exercise many normal DSP data-flow shapes with exactly the same source code for
// float and simd_t<float, N>.
template <typename Sample>
class Dsp
{
public:
    static constexpr size_t FirN = 17;
    static constexpr size_t DelayN = 1024; // power of two for cheap wrapping
    static constexpr size_t FftN = 16;

    Dsp() { reset(); }

    void reset()
    {
        for (auto& x : firHist_) x = splat<Sample>(0.0f);
        for (auto& x : delay_) x = splat<Sample>(0.0f);
        for (auto& x : fftFrame_) x = splat<Sample>(0.0f);
        for (auto& x : combState_) x = splat<Sample>(0.0f);
        for (auto& x : apState_) x = splat<Sample>(0.0f);
        for (auto& x : spectrumHold_) x = splat<Sample>(0.0f);
        z1_ = z2_ = env_ = phase_ = splat<Sample>(0.0f);
        firPos_ = delayPos_ = fftPos_ = 0;
    }

    [[nodiscard]] inline Sample process(Sample x)
    {
        // 1) Input waveshaping + slow modulation: transcendental ops and mixed scalar/vector arithmetic.
        phase_ += splat<Sample>(0.000731f);
        Sample lfo = math_sin(phase_) * 0.5f + math_cos(phase_ * 0.371f) * 0.25f;
        Sample shaped = math_tanh(x * (splat<Sample>(1.15f) + lfo * 0.08f)) + x * 0.06f;

        // 2) Two-pole-ish IIR section with feedback state.
        const Sample iir = shaped * 0.1843f + z1_;
        z1_ = shaped * 0.3686f + z2_ - iir * 0.4771f;
        z2_ = shaped * 0.1843f - iir * 0.2334f;

        // 3) FIR: odd-sized history and non-symmetric coefficients.
        firHist_[firPos_] = iir;
        Sample fir = splat<Sample>(0.0f);
        size_t p = firPos_;
        for (size_t k = 0; k < FirN; ++k)
        {
            fir += firHist_[p] * firCoeffs_[k];
            p = (p == 0 ? FirN - 1 : p - 1);
        }
        if (++firPos_ == FirN) firPos_ = 0;

        // 4) A tiny streaming "FFT-like" analysis. Every FftN samples calculate several real/imag DFT bins.
        // The result is held and fed back into the signal path, so this work cannot be dead-code eliminated.
        fftFrame_[fftPos_++] = fir;
        if (fftPos_ == FftN)
        {
            fftPos_ = 0;
            Sample energy = splat<Sample>(0.0f);
            for (size_t bin = 1; bin <= 5; ++bin)
            {
                Sample re = splat<Sample>(0.0f), im = splat<Sample>(0.0f);
                for (size_t n = 0; n < FftN; ++n)
                {
                    const float a = -6.2831853071795864769f * float(bin * n) / float(FftN);
                    re += fftFrame_[n] * std::cos(a);
                    im += fftFrame_[n] * std::sin(a);
                }
                // Avoid sqrt: magnitude-squared still exercises complex-style MAC structure.
                const Sample mag2 = re * re + im * im;
                spectrumHold_[bin - 1] = mag2 * (1.0f / float(FftN * FftN));
                energy += spectrumHold_[bin - 1] * (0.010f + 0.003f * float(bin));
            }
            env_ = env_ * 0.91f + energy * 0.09f;
        }

        // 5) Modulated multi-tap delay/echo. Fractional interpolation uses two adjacent taps.
        // Delay position is scalar/shared; fractional amount is Sample so lanes can diverge.
        const Sample frac = (lfo + 0.75f) * 0.45f; // approximately [0, 0.45]
        const size_t d0a = (delayPos_ - 149u) & (DelayN - 1u);
        const size_t d0b = (d0a - 1u) & (DelayN - 1u);
        const size_t d1 = (delayPos_ - 337u) & (DelayN - 1u);
        const size_t d2 = (delayPos_ - 701u) & (DelayN - 1u);
        const Sample tap0 = delay_[d0a] + (delay_[d0b] - delay_[d0a]) * frac;
        const Sample echoes = tap0 * 0.43f + delay_[d1] * -0.21f + delay_[d2] * 0.13f;

        // 6) Lightweight Schroeder-ish reverb: 4 comb-like states + 3 all-pass-like stages.
        Sample rv = fir + echoes * 0.37f;
        for (size_t i = 0; i < combState_.size(); ++i)
        {
            const float damp = 0.71f + 0.035f * float(i);
            combState_[i] = combState_[i] * damp + rv * (0.10f + 0.017f * float(i));
            rv += combState_[i] * (0.19f - 0.018f * float(i));
        }
        for (size_t i = 0; i < apState_.size(); ++i)
        {
            const float g = 0.41f + 0.06f * float(i);
            const Sample v = rv - apState_[i] * g;
            rv = apState_[i] + v * g;
            apState_[i] = v;
        }

        // 7) Envelope/nonlinearity branch, abs + tanh. Keep amplitudes bounded for long benchmark runs.
        env_ = env_ * 0.997f + math_abs(rv) * 0.003f;
        Sample out = math_tanh(rv * (splat<Sample>(0.82f) + env_ * 0.04f) + spectrumHold_[2] * 0.002f);

        delay_[delayPos_] = shaped * 0.58f + out * 0.27f + echoes * 0.19f;
        delayPos_ = (delayPos_ + 1u) & (DelayN - 1u);
        return out;
    }

private:
    static constexpr std::array<float, FirN> firCoeffs_ = {
        -0.011f, -0.018f, 0.006f, 0.031f, 0.057f, 0.084f, 0.105f, 0.119f, 0.125f,
         0.113f,  0.091f, 0.062f, 0.032f, 0.009f,-0.005f,-0.011f,-0.008f
    };

    std::array<Sample, FirN> firHist_{};
    std::array<Sample, DelayN> delay_{};
    std::array<Sample, FftN> fftFrame_{};
    std::array<Sample, 4> combState_{};
    std::array<Sample, 3> apState_{};
    std::array<Sample, 5> spectrumHold_{};
    Sample z1_{}, z2_{}, env_{}, phase_{};
    size_t firPos_{}, delayPos_{}, fftPos_{};
};

template<class S> constexpr std::array<float, Dsp<S>::FirN> Dsp<S>::firCoeffs_;

struct Inputs
{
    std::vector<std::array<float, NumProcs>> scalar; // [sample][lane]
    std::vector<SimdF> simd;
};

static Inputs makeInputs()
{
    Inputs r;
    r.scalar.resize(BlockSize);
    r.simd.resize(BlockSize);
    std::array<uint32_t, NumProcs> rng{};
    for (size_t lane = 0; lane < NumProcs; ++lane) rng[lane] = 0x9e3779b9u ^ uint32_t(0x85ebca6bu * (lane + 1));

    for (size_t n = 0; n < BlockSize; ++n)
    {
        std::array<float, NumProcs> pack{};
        for (size_t lane = 0; lane < NumProcs; ++lane)
        {
            rng[lane] = rng[lane] * 1664525u + 1013904223u;
            const float noise = float((rng[lane] >> 9) & 0x7fffffu) * (1.0f / 8388608.0f) - 0.5f;
            const float t = float(n + lane * 29u);
            // Precomputed outside timed region; every lane has distinct deterministic signal.
            pack[lane] = 0.31f * std::sin(t * (0.0113f + 0.0007f * float(lane)))
                + 0.17f * std::cos(t * (0.0271f + 0.0003f * float(lane)))
                + 0.08f * noise;
        }
        r.scalar[n] = pack;
        r.simd[n] = SimdF(pack);
    }
    return r;
}

struct ScalarRun
{
    double seconds{};
    std::array<float, NumProcs> last{};
    double checksum{};
};
struct SimdRun
{
    double seconds{};
    SimdF last{};
    double checksum{};
};

static ScalarRun runScalar(const Inputs& in)
{
    Dsp<float> procSingle[NumProcs];
    std::array<float, NumProcs> last{};
    double checksum = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < Passes; ++pass)
    {
        for (size_t n = 0; n < BlockSize; ++n)
        {
            for (size_t lane = 0; lane < NumProcs; ++lane)
                last[lane] = procSingle[lane].process(in.scalar[n][lane]);
        }
        // One tiny observable reduction per block; identical logical amount for both paths.
        for (size_t lane = 0; lane < NumProcs; ++lane)
            checksum += double(last[lane]) * double(lane + 1);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return { std::chrono::duration<double>(t1 - t0).count(), last, checksum };
}

static SimdRun runSimd(const Inputs& in)
{
    Dsp<SimdF> procSimd;
    SimdF last(0.0f);
    double checksum = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int pass = 0; pass < Passes; ++pass)
    {
        for (size_t n = 0; n < BlockSize; ++n)
            last = procSimd.process(in.simd[n]);

        for (size_t lane = 0; lane < NumProcs; ++lane)
            checksum += double(last.dat[lane]) * double(lane + 1);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return { std::chrono::duration<double>(t1 - t0).count(), last, checksum };
}


struct StreamAccuracy
{
    float maxAbs{};
    float maxRel{};
    double rms{};
    size_t worstSample{};
    size_t worstLane{};
    uint64_t compared{};
};

static StreamAccuracy validateNumerics(const Inputs& in)
{
    Dsp<float> scalar[NumProcs];
    Dsp<SimdF> simd;
    StreamAccuracy q{};
    double ss = 0.0;

    // Full-stream validation, untimed. Two blocks are enough to wrap the 1024-sample delay many times
    // and execute all FIR/IIR/FFT/reverb state paths repeatedly.
    constexpr int ValidationPasses = 2;
    for (int pass = 0; pass < ValidationPasses; ++pass)
    {
        for (size_t n = 0; n < BlockSize; ++n)
        {
            std::array<float, NumProcs> ys{};
            for (size_t lane = 0; lane < NumProcs; ++lane)
                ys[lane] = scalar[lane].process(in.scalar[n][lane]);
            const SimdF yv = simd.process(in.simd[n]);

            for (size_t lane = 0; lane < NumProcs; ++lane)
            {
                const float ae = std::abs(ys[lane] - yv.dat[lane]);
                const float den = std::max({ 1.0e-8f, std::abs(ys[lane]), std::abs(yv.dat[lane]) });
                const float re = ae / den;
                if (ae > q.maxAbs)
                {
                    q.maxAbs = ae;
                    q.worstSample = size_t(pass) * BlockSize + n;
                    q.worstLane = lane;
                }
                q.maxRel = std::max(q.maxRel, re);
                ss += double(ae) * double(ae);
                ++q.compared;
            }
        }
    }
    q.rms = std::sqrt(ss / double(q.compared));
    return q;
}

struct Accuracy
{
    float maxAbs{};
    float maxRel{};
    double rms{};
    size_t worstLane{};
};

static Accuracy compareFinal(const std::array<float, NumProcs>& a, const SimdF& b)
{
    Accuracy q{};
    double ss = 0.0;
    for (size_t lane = 0; lane < NumProcs; ++lane)
    {
        const float ae = std::abs(a[lane] - b.dat[lane]);
        const float den = std::max({ 1.0e-8f, std::abs(a[lane]), std::abs(b.dat[lane]) });
        const float re = ae / den;
        if (ae > q.maxAbs) { q.maxAbs = ae; q.worstLane = lane; }
        q.maxRel = std::max(q.maxRel, re);
        ss += double(ae) * double(ae);
    }
    q.rms = std::sqrt(ss / double(NumProcs));
    return q;
}

int main()
{
    std::cout << "SimdHelper evaluation\n"
        << "  lanes / NumProcs : " << NumProcs << '\n'
        << "  block samples    : " << BlockSize << '\n'
        << "  passes           : " << Passes << '\n'
        << "  samples/lane     : " << (uint64_t(BlockSize) * uint64_t(Passes)) << '\n'
        << "  total lane-samps : " << (uint64_t(BlockSize) * uint64_t(Passes) * NumProcs) << "\n\n";

    const Inputs inputs = makeInputs();
    const StreamAccuracy streamAcc = validateNumerics(inputs);

    // Untimed warm-up also validates that both instantiations compile and execute before timing.
    {
        Dsp<float> s[NumProcs];
        Dsp<SimdF> v;
        for (size_t n = 0; n < std::min<size_t>(BlockSize, 128); ++n)
        {
            for (size_t lane = 0; lane < NumProcs; ++lane) (void)s[lane].process(inputs.scalar[n][lane]);
            (void)v.process(inputs.simd[n]);
        }
    }

    const ScalarRun scalar = runScalar(inputs);
    const SimdRun simd = runSimd(inputs);
    const Accuracy acc = compareFinal(scalar.last, simd.last);

    const double scalarM = double(BlockSize) * double(Passes) * double(NumProcs) / scalar.seconds / 1.0e6;
    const double simdM = double(BlockSize) * double(Passes) * double(NumProcs) / simd.seconds / 1.0e6;

    std::cout << std::fixed << std::setprecision(6)
        << "scalar Dsp<float>[" << NumProcs << "] : " << scalar.seconds << " s  (" << scalarM << " M lane-samples/s)\n"
        << "simd   Dsp<simd_t>       : " << simd.seconds << " s  (" << simdM << " M lane-samples/s)\n"
        << "speedup                    : " << (scalar.seconds / simd.seconds) << " x\n\n"
        << std::scientific
        << "stream max abs error       : " << streamAcc.maxAbs << " (sample " << streamAcc.worstSample << ", lane " << streamAcc.worstLane << ")\n"
        << "stream max rel error       : " << streamAcc.maxRel << '\n'
        << "stream RMS error           : " << streamAcc.rms << '\n'
        << "numeric max abs error      : " << acc.maxAbs << " (lane " << acc.worstLane << ")\n"
        << "numeric max rel error      : " << acc.maxRel << '\n'
        << "numeric RMS error          : " << acc.rms << '\n'
        << "scalar checksum            : " << scalar.checksum << '\n'
        << "simd checksum              : " << simd.checksum << '\n';

    // Loose enough for differing contraction/vector math choices, strict enough to catch state/layout bugs.
    const bool ok = std::isfinite(scalar.checksum) && std::isfinite(simd.checksum)
        && streamAcc.maxAbs < 2.0e-4f && streamAcc.maxRel < 2.0e-3f
        && acc.maxAbs < 2.0e-4f && acc.maxRel < 2.0e-3f;
    std::cout << "\nRESULT: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 2;
}
