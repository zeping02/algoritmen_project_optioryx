#include "optioryx/algorithms/seed_selection_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace optioryx {

namespace {

// Kies het startpunt van een nieuwe batch. De twee snake-spanregels zijn de
// eendimensionale adaptaties van respectievelijk smallest en greatest covering area.
// highest_standalone_cost blijft beschikbaar voor bestaande configuraties.
int pick_seed(
    const BatchEvaluator& evaluator,
    int wave_index,
    const std::vector<int>& remaining_orders,
    SeedRule rule) {
    int best_order = remaining_orders.front();
    const bool minimize = rule == SeedRule::smallest_snake_span;
    double best_score = minimize ? std::numeric_limits<double>::infinity()
                                 : -std::numeric_limits<double>::infinity();

    for (const int order_index : remaining_orders) {
        double score = 0.0;
        switch (rule) {
            case SeedRule::highest_standalone_cost:
                score = evaluator.batch_cost(wave_index, {order_index});
                break;
            case SeedRule::largest_snake_span:
            case SeedRule::smallest_snake_span:
                score = static_cast<double>(evaluator.order_span_width(order_index));
                break;
        }
        if ((minimize && score < best_score) || (!minimize && score > best_score)) {
            best_score = score;
            best_order = order_index;
        }
    }

    return best_order;
}

struct CandidateScore {
    int order_index = -1;
    double incremental_cost = 0.0;
    double span_expansion = 0.0;
    std::size_t location_overlap = 0U;
};

bool lower_immediate_cost(const CandidateScore& left, const CandidateScore& right) {
    if (left.incremental_cost != right.incremental_cost) {
        return left.incremental_cost < right.incremental_cost;
    }
    return left.span_expansion < right.span_expansion;
}

int pick_candidate(
    const BatchEvaluator& evaluator,
    int wave_index,
    const std::vector<int>& current_orders,
    const std::vector<int>& remaining_orders,
    std::size_t batch_capacity,
    const SeedSelectionConfig& config) {
    // Kies welke resterende order aan de huidige batch wordt toegevoegd.
    // We berekenen per kandidaat extra routekost, extra snake-span en het aantal
    // unieke picklocaties dat al door de huidige batch wordt afgedekt.

    std::vector<CandidateScore> scores;
    scores.reserve(remaining_orders.size());
    const BatchStats current_stats = evaluator.batch_stats(wave_index, current_orders);
    const double current_cost = current_stats.cost;

    for (const int candidate_order : remaining_orders) {
        std::vector<int> trial_orders = current_orders;
        trial_orders.push_back(candidate_order);
        const BatchStats trial_stats = evaluator.batch_stats(wave_index, trial_orders);
        const BatchStats candidate_stats = evaluator.batch_stats(wave_index, {candidate_order});

        CandidateScore score;
        score.order_index = candidate_order;
        score.incremental_cost = trial_stats.cost - current_cost;
        score.span_expansion = static_cast<double>(trial_stats.span_width - current_stats.span_width);
        score.location_overlap =
            current_stats.unique_item_count + candidate_stats.unique_item_count - trial_stats.unique_item_count;
        scores.push_back(score);
    }

    if (config.selection_rule == SelectionRule::minimum_incremental_cost) {
        // Zuiver kostencriterium: kies de order die de batch het minst duurder maakt.
        return std::min_element(scores.begin(), scores.end(), lower_immediate_cost)
            ->order_index;
    }

    if (config.selection_rule == SelectionRule::minimum_span_expansion) {
        // 1D-adaptatie van smallest additional covering area: vergroot het
        // interval tussen de uiterste snake-ranks zo weinig mogelijk.
        return std::min_element(scores.begin(), scores.end(), [](const CandidateScore& left, const CandidateScore& right) {
                   if (left.span_expansion != right.span_expansion) {
                       return left.span_expansion < right.span_expansion;
                   }
                   return left.incremental_cost < right.incremental_cost;
               })
            ->order_index;
    }

    if (config.selection_rule == SelectionRule::maximum_location_overlap) {
        // Adaptatie van greatest identical/overlap locations: maximaliseer de
        // doorsnede van unieke picklocaties en breek ties op directe routekost.
        return std::min_element(scores.begin(), scores.end(), [](const CandidateScore& left, const CandidateScore& right) {
                   if (left.location_overlap != right.location_overlap) {
                       return left.location_overlap > right.location_overlap;
                   }
                   return lower_immediate_cost(left, right);
               })
            ->order_index;
    }

    if (config.selection_rule == SelectionRule::bounded_two_step_lookahead) {
        // Beperk alleen de eerste vertakking tot de K beste directe increments.
        // Voor elk van die kandidaten wordt de beste mogelijke tweede toevoeging
        // volledig bekeken. Zo blijft de stap O(K*n), terwijl de keuze echt twee
        // stappen vooruitkijkt in plaats van twee losse scores te mengen.
        std::stable_sort(scores.begin(), scores.end(), lower_immediate_cost);
        const std::size_t shortlist_size = std::min(
            scores.size(),
            static_cast<std::size_t>(config.lookahead_width));
        scores.resize(shortlist_size);

        const bool has_capacity_for_second = current_orders.size() + 2U <= batch_capacity;
        if (!has_capacity_for_second || remaining_orders.size() < 2U) {
            return scores.front().order_index;
        }

        const CandidateScore* best_first = nullptr;
        double best_two_step_cost = std::numeric_limits<double>::infinity();
        for (const CandidateScore& first : scores) {
            std::vector<int> two_step_orders = current_orders;
            two_step_orders.push_back(first.order_index);

            double first_best_two_step_cost = std::numeric_limits<double>::infinity();
            for (const int second_order : remaining_orders) {
                if (second_order == first.order_index) {
                    continue;
                }
                two_step_orders.push_back(second_order);
                first_best_two_step_cost = std::min(
                    first_best_two_step_cost,
                    evaluator.batch_cost(wave_index, two_step_orders));
                two_step_orders.pop_back();
            }

            if (best_first == nullptr || first_best_two_step_cost < best_two_step_cost ||
                (first_best_two_step_cost == best_two_step_cost && lower_immediate_cost(first, *best_first))) {
                best_first = &first;
                best_two_step_cost = first_best_two_step_cost;
            }
        }
        return best_first->order_index;
    }

    if (config.selection_rule != SelectionRule::hybrid_score) {
        throw std::logic_error("Unsupported seed-selection rule");
    }

    // Legacy weighted hybrid remains executable for old configs, but is no
    // longer part of the paper factorial suite.
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
        // Hybrid combineert genormaliseerde routekost en span-uitbreiding.
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
    // Baseline-algoritme: bouw batches een voor een op rond een seed-order.
    // Het is snel en goed uitlegbaar, maar mist de latere routeherbouw van de greedy-eindsolver.
    const auto started = std::chrono::steady_clock::now();
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));

    if (config.selection_rule == SelectionRule::bounded_two_step_lookahead && config.lookahead_width <= 0) {
        throw std::invalid_argument("lookahead_width must be positive for bounded_two_step_lookahead");
    }

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
                pick_candidate(evaluator, wave_index, batch.order_indices, remaining_orders, instance.nmax, config);
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
    result.metadata["paper_adaptation"] =
        "covering_area_to_snake_span;identical_locations_to_unique_location_overlap";
    if (config.selection_rule == SelectionRule::bounded_two_step_lookahead) {
        result.metadata["lookahead_width"] = std::to_string(config.lookahead_width);
    }

    const auto finished = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

}  // namespace optioryx
