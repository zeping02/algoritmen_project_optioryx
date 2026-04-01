#pragma once

#include "optioryx/core/evaluator.hpp"
#include "optioryx/core/model.hpp"

namespace optioryx {

WaveMetrics compute_wave_metrics(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index);

}  // namespace optioryx
