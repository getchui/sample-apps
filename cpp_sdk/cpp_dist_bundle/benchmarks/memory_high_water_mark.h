#pragma once

namespace Trueface {
namespace Benchmarks {

class MemoryHighWaterMarkTracker {
public:
    MemoryHighWaterMarkTracker();

    void resetVmHighWaterMark();
    double getVmHighWaterMark() const;
    double getDifferenceFromBaseline() const;

private:
    double m_baseline;
};

} // namespace Benchmarks
} // namespace Trueface
