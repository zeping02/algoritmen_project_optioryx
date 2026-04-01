#include "optioryx/core/analysis.hpp"

#include <cmath>
#include <numeric>
#include <unordered_set>

namespace optioryx {

namespace {

double jaccard_similarity(const Order& left, const Order& right) {
    std::unordered_set<int> left_items(left.items.begin(), left.items.end());
    std::unordered_set<int> right_items(right.items.begin(), right.items.end());

    std::size_t intersection_size = 0;
    for (const int item : left_items) {
        if (right_items.contains(item)) {
            ++intersection_size;
        }
    }

    const std::size_t union_size = left_items.size() + right_items.size() - intersection_size;
    if (union_size == 0U) {
        return 0.0;
    }
    return static_cast<double>(intersection_size) / static_cast<double>(union_size);
}

}  // namespace

WaveMetrics compute_wave_metrics(const ProblemInstance& instance, const BatchEvaluator& evaluator, int wave_index) {
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));
    WaveMetrics metrics;
    if (wave.order_indices.size() < 2U) {
        return metrics;
    }

    double overlap_sum = 0.0;
    int pair_count = 0;
    std::vector<double> spans;
    spans.reserve(wave.order_indices.size());

    for (std::size_t left = 0; left < wave.order_indices.size(); ++left) {
        spans.push_back(static_cast<double>(evaluator.order_span_width(wave.order_indices[left])));
        for (std::size_t right = left + 1; right < wave.order_indices.size(); ++right) {
            overlap_sum += jaccard_similarity(
                instance.orders.at(static_cast<std::size_t>(wave.order_indices[left])),
                instance.orders.at(static_cast<std::size_t>(wave.order_indices[right])));
            ++pair_count;
        }
    }

    metrics.overlap_density = pair_count == 0 ? 0.0 : overlap_sum / static_cast<double>(pair_count);

    const double mean_span = std::accumulate(spans.begin(), spans.end(), 0.0) / static_cast<double>(spans.size());
    double variance = 0.0;
    for (const double span : spans) {
        const double delta = span - mean_span;
        variance += delta * delta;
    }
    variance /= static_cast<double>(spans.size());
    metrics.span_dispersion = std::sqrt(variance);

    return metrics;
}

}  // namespace optioryx
