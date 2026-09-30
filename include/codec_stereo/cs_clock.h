#ifndef CODEC_STEREO_CS_CLOCK_H
#define CODEC_STEREO_CS_CLOCK_H

/* Monotonic millisecond clock, portable across MSVC, MinGW-w64 and POSIX.
   clock_gettime(CLOCK_MONOTONIC) has no MSVC declaration, so Windows uses
   QueryPerformanceCounter (available in both MSVC and MinGW). */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static inline double cs_now_ms(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <time.h>

static inline double cs_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

#endif
