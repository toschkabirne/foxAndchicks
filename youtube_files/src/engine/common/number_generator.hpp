#pragma once
#include <random>


class CustomNumberGenerator
{
protected:
    uint64_t s[2];

    CustomNumberGenerator()
    {
        setSeed(1);
    }

public:
    void setSeed(uint64_t seed_)
    {
        uint64_t constexpr init_shift = 16;
        if (seed_ == 0) {
            std::cout << "[WARNING] Seed cannot be equal to 0, using 1" << std::endl;
            s[0] = 1 << (init_shift / 2);
            s[1] = 1 << init_shift;
        } else {
            s[0] = seed_ << (init_shift / 2);
            s[1] = seed_ << init_shift;
        }
    }

    uint64_t getNext()
    {
        uint64_t x = s[0];
        uint64_t const y =  s[1];
        s[0] = y;
        x ^= x << 23; // shift & xor
        x ^= x >> 17; // shift & xor
        x ^= y; // xor
        s[1] = x + y;
        return x;
    }

    double getNextDouble()
    {
        return static_cast<double>(getNext()) / static_cast<double>(std::numeric_limits<uint64_t>::max());
    }
};


template<typename T>
class RealNumberGenerator : public CustomNumberGenerator
{
public:
    RealNumberGenerator()
        : CustomNumberGenerator()
    {}

    T get()
    {
        return static_cast<T>(getNextDouble());
    }

    T getUnder(T max)
    {
        T const v = get();
        //std::cout << v << std::endl;
        return v * max;
    }

    T getRange(T min, T max)
    {
        return min + get() * (max - min);
    }

    T getRange(T width)
    {
        return getRange(-width * 0.5f, width * 0.5f);
    }

    T getFullRange(T width)
    {
        return getRange(static_cast<T>(2.0) * width);
    }

    [[nodiscard]]
    bool proba(T threshold)
    {
        return get() < threshold;
    }
};

/*
template<typename T>
class RNG
{
private:
    static RealNumberGenerator<T> gen;

public:
    static T get()
    {
        return gen.get();
    }

    static float getUnder(T max)
    {
        return gen.getUnder(max);
    }

    static uint64_t getUintUnder(uint64_t max)
    {
        return static_cast<uint64_t>(gen.getUnder(static_cast<float>(max) + 1.0f));
    }

    static float getRange(T min, T max)
    {
        return gen.getRange(min, max);
    }

    static float getRange(T width)
    {
        return gen.getRange(width);
    }

    static float getFullRange(T width)
    {
        return gen.getRange(static_cast<T>(2.0f) * width);
    }

    static bool proba(float threshold)
    {
        return get() < threshold;
    }

    static void setSeed(uint32_t seed)
    {
        gen.setSeed(seed);
    }
};

using RNGf = RNG<float>;

template<typename T>
RealNumberGenerator<T> RNG<T>::gen = RealNumberGenerator<T>();
*/
