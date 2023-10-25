#pragma once

namespace Trueface {
namespace Benchmarks {

class MemoryHighWaterMarkTracker {
public:
    MemoryHighWaterMarkTracker();

    void resetVmHighWaterMark();
    float getVmHighWaterMark() const;
    float getDifferenceFromBaseline() const;

private:
    float m_baseline;
};

} // namespace Benchmarks
} // namespace Trueface
