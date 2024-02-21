#include "memory_high_water_mark.h"

#include <fstream>
#if !defined(WIN32) && !defined(_WIN32)
#include <unistd.h>
#ifdef __APPLE__
#include <sys/time.h>
#endif
#include <sys/resource.h>
#else
#include<windows.h>
#endif


using namespace Trueface::Benchmarks;

MemoryHighWaterMarkTracker::MemoryHighWaterMarkTracker() {
    resetVmHighWaterMark();
    m_baseline = getVmHighWaterMark();
}

void MemoryHighWaterMarkTracker::resetVmHighWaterMark() {
#if !defined(WIN32) && !defined(_WIN32)
#ifndef __APPLE__
    //
    // LINUX ONLY: https://www.kernel.org/doc/html/latest/filesystems/proc.html
    // "To reset the peak resident set size ("high water mark") to the process's current value:"
    // > echo 5 > /proc/PID/clear_refs
    //
    std::ofstream clear_refs_stream{"/proc/self/clear_refs"};
    clear_refs_stream << "5" << std::endl;
    clear_refs_stream.close();
    ::sleep(1);
    m_baseline = getVmHighWaterMark();
#endif
#endif
}

float MemoryHighWaterMarkTracker::getVmHighWaterMark() const {
#if !defined(WIN32) && !defined(_WIN32)
    struct rusage usage {};
    if (!getrusage(RUSAGE_SELF, &usage)) {
        //
        // https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/getrusage.2.html
        // https://man7.org/linux/man-pages/man2/getrusage.2.html
        //
        // both Linux MacOS return ru_maxrss in kilobytes, but we convert to megabytes
        //
        return usage.ru_maxrss / 1000.f;
    }
#endif
    return 0.0;
}

float MemoryHighWaterMarkTracker::getDifferenceFromBaseline() const {
    auto current = getVmHighWaterMark();

    return m_baseline < current ? current - m_baseline : 0.0f;
}