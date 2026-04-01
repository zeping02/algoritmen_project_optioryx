#include "optioryx/algorithms/seed_selection_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace optioryx {

namespace {

int pick_seed(
    const BatchEvaluator& evaluator,
    int wave_index,
    const std::vector<int>& remaining_orders,
    SeedRule rule) {
    int best_order = remaining_orders.front();
    double best_score = -std::numeric_limits<double>::infinity();

    for (const int order_index : remaining_orders) {
        double score = 0.0;
        switch (rule) {
            case SeedRule::highest_standalone_cost:
                score = evaluator.batch_cost(wave_index, {order_index});
                break;
            case SeedRule::largest_snake_span:
                score = static_cast<double>(evaluator.order_span_width(order_index));
                break;
        }
        if (score > best_score) {
            best_score = score;
            best_order = order_index;
        }
    }

    return best_order;
}

int pick_candidate(
    const BatchEvaluator& evaluator,
    int wave_index,
    const std::vector<int>& current_orders,
    const std::vector<int>& remaining_orders,
    SelectionRule rule) {
    struct CandidateScore {
        int order_index = -1;
        double incremental_cost = 0.0;
        double span_expansion = 0.0;
    };

    std::vector<CandidateScore> scores;
    scores.reserve(remaining_orders.size());
    const BatchStats current_stats = evaluator.batch_stats(wave_index, current_orders);
    const double current_cost = current_stats.cost;

    for (const int candidate_order : remaining_orders) {
        std::vector<int> trial_orders = current_orders;
        trial_orders.push_back(candidate_order);
        const BatchStats trial_stats = evaluator.batch_stats(wave_index, trial_orders);

        CandidateScore score;
        score.order_index = candidate_order;
        score.incremental_cost = trial_stats.cost - current_cost;
        score.span_expansion = static_cast<double>(trial_stats.span_width - current_stats.span_width);
        scores.push_back(score);
    }

    if (rule == SelectionRule::minimum_incremental_cost) {
        return std::min_element(scores.begin(), scores.end(), [](const CandidateScore& left, const CandidateScore& right) {
                   if (left.incremental_cost != right.incremental_cost) {
                       return left.incremental_cost < right.incremental_cost;
                   }
                   return left.span_expansion < right.span_expansion;
               })
            ->order_index;
    }

    if (rule == SelectionRule::minimum_span_expansion) {
        return std::min_element(scores.begin(), scores.end(), [](const CandidateScore& left, const CandidateScore& right) {
                   if (left.span_expansion != right.span_expansion) {
                       return left.span_expansion < right.span_expansion;
                   }
                   return left.incremental_cost < right.incremental_cost;
               })
            ->order_index;
    }

    double min_incremental = scores.front().incremental_cost;
    double max_incremental = scores.front().incremental_cost;
    double min_span = scores.front().span_expansion;
    double max_span = scores.front().span_expansion;
    for (const CandidateScore& score : scores) {
        min_incremental = std::min(min_incremental, score.incremental_cost);
        max_incremental = std::max(max_incremental, score.incremental_cost);
        min_span = std::min(min_span, score.span_expansion);
        max_span = std::max(max_span, score.span_expansion);
    }

    auto normalize = [](double value, double minimum, double maximum) {
        if (std::abs(maximum - minimum) <= 1e-12) {
            return 0.0;
        }
        return (value - minimum) / (maximum - minimum);
    };

    int best_order = scores.front().order_index;
    double best_value = std::numeric_limits<double>::infinity();
    for (const CandidateScore& score : scores) {
        const double objective = 0.7 * normalize(score.incremental_cost, min_incremental, max_incremental) +
                                 0.3 * normalize(score.span_expansion, min_span, max_span);
        if (objective < best_value) {
            best_value = objective;
            best_order = score.order_index;
        }
    }

    return best_order;
}

}  // namespace

WaveSolveResult solve_wave_seed_selection(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const SeedSelectionConfig& config) {
    const auto started = std::chrono::steady_clock::now();
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));

    std::vector<int> remaining_orders = wave.order_indices;
    std::vector<Batch> batches;

    while (!remaining_orders.empty()) {
        const int seed_order = pick_seed(evaluator, wave_index, remaining_orders, config.seed_rule);
        remaining_orders.erase(std::remove(remaining_orders.begin(), remaining_orders.end(), seed_order), remaining_orders.end());

        Batch batch;
        batch.wave_id = wave.id;
        batch.order_indices.push_back(seed_order);

        while (batch.order_indices.size() < instance.nmax && !remaining_orders.empty()) {
            const int next_order =
                pick_candidate(evaluator, wave_index, batch.order_indices, remaining_orders, config.selection_rule);
            batch.order_indices.push_back(next_order);
            remaining_orders.erase(std::remove(remaining_orders.begin(), remaining_orders.end(), next_order), remaining_orders.end());
        }

        batches.push_back(batch);
    }

    WaveSolveResult result;
    result.batches = std::move(batches);
    result.objective = evaluator.wave_cost(wave_index, result.batches);
    result.metadata["seed_rule"] = to_string(config.seed_rule);
    result.metadata["selection_rule"] = to_string(config.selection_rule);
    result.metadata["config_id"] = config.config_id;
    result.metadata["paper_adaptation"] = "distance_area_style_rules_adapted_to_snake_route";

    const auto finished = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

}  // namespace optioryx
