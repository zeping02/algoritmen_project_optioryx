#include "optioryx/algorithms/common.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace optioryx {

namespace {

// Configbestanden mogen leesbaar blijven: hoofdletters, koppeltekens en
// underscores worden hier genormaliseerd naar dezelfde interne naam.
std::string normalize(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        if (character == '-') {
            return '_';
        }
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

}  // namespace

StartSolutionStrategy parse_start_strategy(const std::string& value) {
    const std::string normalized = normalize(value);
    if (normalized == "greedy") {
        return StartSolutionStrategy::greedy;
    }
    if (normalized == "randomized" || normalized == "random") {
        return StartSolutionStrategy::randomized;
    }
    throw std::runtime_error("Unknown start strategy: " + value);
}

// De parse-functies vertalen tekst uit .cfg-bestanden naar enumwaarden.
// Onbekende waarden geven bewust een fout, zodat typfouten in experimenten
// niet stilzwijgend een andere solverconfiguratie opleveren.
NeighborhoodStrategy parse_neighborhood_strategy(const std::string& value) {
    const std::string normalized = normalize(value);
    if (normalized == "relocate") {
        return NeighborhoodStrategy::relocate;
    }
    if (normalized == "swap") {
        return NeighborhoodStrategy::swap;
    }
    if (normalized == "merge_split" || normalized == "mergesplit" || normalized == "merge-split") {
        return NeighborhoodStrategy::merge_split;
    }
    if (normalized == "mixed") {
        return NeighborhoodStrategy::mixed;
    }
    throw std::runtime_error("Unknown neighborhood strategy: " + value);
}

CoolingSchedule parse_cooling_schedule(const std::string& value) {
    const std::string normalized = normalize(value);
    if (normalized == "geometric") {
        return CoolingSchedule::geometric;
    }
    if (normalized == "lundy_mees" || normalized == "lundymees") {
        return CoolingSchedule::lundy_mees;
    }
    throw std::runtime_error("Unknown cooling schedule: " + value);
}

SeedRule parse_seed_rule(const std::string& value) {
    const std::string normalized = normalize(value);
    if (normalized == "highest_standalone_cost" || normalized == "cost") {
        return SeedRule::highest_standalone_cost;
    }
    if (normalized == "largest_snake_span" || normalized == "span") {
        return SeedRule::largest_snake_span;
    }
    throw std::runtime_error("Unknown seed rule: " + value);
}

SelectionRule parse_selection_rule(const std::string& value) {
    const std::string normalized = normalize(value);
    if (normalized == "minimum_incremental_cost" || normalized == "incremental_cost") {
        return SelectionRule::minimum_incremental_cost;
    }
    if (normalized == "minimum_span_expansion" || normalized == "span_expansion") {
        return SelectionRule::minimum_span_expansion;
    }
    if (normalized == "hybrid_score" || normalized == "hybrid") {
        return SelectionRule::hybrid_score;
    }
    throw std::runtime_error("Unknown selection rule: " + value);
}

std::string to_string(StartSolutionStrategy value) {
    // De to_string-functies worden gebruikt voor metadata, benchmark-CSV's en JSON-output.
    switch (value) {
        case StartSolutionStrategy::greedy:
            return "greedy";
        case StartSolutionStrategy::randomized:
            return "randomized";
    }
    return "unknown";
}

std::string to_string(NeighborhoodStrategy value) {
    switch (value) {
        case NeighborhoodStrategy::relocate:
            return "relocate";
        case NeighborhoodStrategy::swap:
            return "swap";
        case NeighborhoodStrategy::merge_split:
            return "merge_split";
        case NeighborhoodStrategy::mixed:
            return "mixed";
    }
    return "unknown";
}

std::string to_string(CoolingSchedule value) {
    switch (value) {
        case CoolingSchedule::geometric:
            return "geometric";
        case CoolingSchedule::lundy_mees:
            return "lundy_mees";
    }
    return "unknown";
}

std::string to_string(SeedRule value) {
    switch (value) {
        case SeedRule::highest_standalone_cost:
            return "highest_standalone_cost";
        case SeedRule::largest_snake_span:
            return "largest_snake_span";
    }
    return "unknown";
}

std::string to_string(SelectionRule value) {
    switch (value) {
        case SelectionRule::minimum_incremental_cost:
            return "minimum_incremental_cost";
        case SelectionRule::minimum_span_expansion:
            return "minimum_span_expansion";
        case SelectionRule::hybrid_score:
            return "hybrid_score";
    }
    return "unknown";
}

}  // namespace optioryx
