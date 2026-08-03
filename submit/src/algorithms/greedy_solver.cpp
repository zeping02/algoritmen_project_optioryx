#include "optioryx/algorithms/greedy_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace optioryx {

namespace {

// Interne route tijdens de savings-constructie.
// item_mask maakt het goedkoop om de kost van een merge te berekenen:
// de routekost hangt af van de unieke picklocaties, niet van het aantal orders.
struct ActiveBatch {
    Batch batch;
    std::vector<std::uint64_t> item_mask;
    double cost = 0.0;
    int version = 0;
    bool active = true;
};

// Zoekrepresentatie voor local search en rebuilds.
// item_counts bewaart aantallen in plaats van bits, omdat orders tijdelijk
// verwijderd en opnieuw toegevoegd worden. Als meerdere orders dezelfde locatie
// gebruiken, blijft die locatie aanwezig zolang de count groter dan nul is.
struct SearchBatch {
    Batch batch;
    std::vector<std::uint16_t> item_counts;
    double cost = 0.0;
    int min_rank = 0;
    int max_rank = 0;
};

// Kandidaatmerge voor de priority queue.
// De version-velden maken lazy invalidation mogelijk: oude kandidaten blijven
// in de queue staan, maar worden genegeerd zodra een van hun routes gewijzigd is.
struct MergeCandidate {
    double saving = 0.0;
    double merged_cost = 0.0;
    std::size_t left_id = 0;
    std::size_t right_id = 0;
    int left_version = 0;
    int right_version = 0;
};

enum class LocalActionType {
    none,
    swap_orders,
    move_left_to_right,
    move_right_to_left,
};

struct LocalAction {
    LocalActionType type = LocalActionType::none;
    std::size_t left_pos = 0;
    std::size_t right_pos = 0;
    double delta = 0.0;
};

struct TrialDestination {
    std::size_t batch_index = 0;
    std::vector<std::uint16_t> item_counts;
    std::size_t order_count = 0;
    double cost = 0.0;
    std::vector<int> assigned_orders;
};

struct EliminationState {
    std::vector<TrialDestination> destinations;
    double incremental_cost = 0.0;
};

struct DestinationChoice {
    std::size_t destination_index = 0;
    double incremental_cost = 0.0;
    double new_cost = 0.0;
};

struct EliminationPlan {
    std::size_t source_index = 0;
    std::vector<TrialDestination> destinations;
    double saving = 0.0;
};

struct MergeCandidateLess {
    bool operator()(const MergeCandidate& left, const MergeCandidate& right) const {
        // priority_queue geeft de "grootste" kandidaat eerst terug.
        // Eerst maximaliseren we saving, daarna kiezen we bij gelijke saving
        // de goedkoopste merged route als stabiele tie-breaker.
        if (std::abs(left.saving - right.saving) > 1e-9) {
            return left.saving < right.saving;
        }
        if (std::abs(left.merged_cost - right.merged_cost) > 1e-9) {
            return left.merged_cost > right.merged_cost;
        }
        return std::tie(left.left_id, left.right_id) > std::tie(right.left_id, right.right_id);
    }
};

// Een vaste sortering van locaties volgens de snake-route.
// Alle routekosten lopen daarna lineair door deze lijst.
std::vector<int> build_items_by_snake_rank(const ProblemInstance& instance) {
    std::vector<int> items(instance.item_ids.size());
    std::iota(items.begin(), items.end(), 0);
    std::sort(items.begin(), items.end(), [&](const int left, const int right) {
        return instance.snake_rank.at(static_cast<std::size_t>(left)) <
               instance.snake_rank.at(static_cast<std::size_t>(right));
    });
    return items;
}

// Bitmask van de locaties die door een order nodig zijn.
// START en END worden niet in het masker gezet: die worden apart bij elke route
// aan het begin en einde toegevoegd.
std::vector<std::uint64_t> build_order_mask(
    const ProblemInstance& instance,
    int order_index,
    std::size_t mask_words) {
    std::vector<std::uint64_t> mask(mask_words, 0U);
    const Order& order = instance.orders.at(static_cast<std::size_t>(order_index));
    for (const int item : order.items) {
        if (item == instance.start_index || item == instance.end_index) {
            continue;
        }
        const std::size_t item_index = static_cast<std::size_t>(item);
        mask[item_index / 64U] |= (std::uint64_t{1} << (item_index % 64U));
    }
    return mask;
}

// Voor local search is een lijst per order handiger dan alleen een bitmask,
// omdat een order snel uit item_counts verwijderd en toegevoegd moet worden.
std::vector<std::vector<int>> build_order_items_by_snake_rank(const ProblemInstance& instance) {
    std::vector<std::vector<int>> order_items(instance.orders.size());
    for (std::size_t order_index = 0; order_index < instance.orders.size(); ++order_index) {
        std::vector<int> items = instance.orders[order_index].items;
        items.erase(std::remove(items.begin(), items.end(), instance.start_index), items.end());
        items.erase(std::remove(items.begin(), items.end(), instance.end_index), items.end());
        std::sort(items.begin(), items.end());
        items.erase(std::unique(items.begin(), items.end()), items.end());
        std::sort(items.begin(), items.end(), [&](const int left, const int right) {
            return instance.snake_rank.at(static_cast<std::size_t>(left)) <
                   instance.snake_rank.at(static_cast<std::size_t>(right));
        });
        order_items[order_index] = std::move(items);
    }
    return order_items;
}

bool mask_contains(const std::vector<std::uint64_t>& mask, int item) {
    const std::size_t item_index = static_cast<std::size_t>(item);
    return (mask[item_index / 64U] & (std::uint64_t{1} << (item_index % 64U))) != 0U;
}

double route_cost_for_mask(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::uint64_t>& mask) {
    double cost = 0.0;
    int previous = instance.start_index;
    bool has_items = false;
    // Score van een route: bezoek elke unieke locatie eenmaal in snake-volgorde.
    for (const int item : items_by_snake_rank) {
        if (!mask_contains(mask, item)) {
            continue;
        }
        cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
        previous = item;
        has_items = true;
    }
    if (!has_items) {
        return 0.0;
    }
    cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance.end_index));
    return cost;
}

double route_cost_for_counts(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::uint16_t>& item_counts) {
    double cost = 0.0;
    int previous = instance.start_index;
    bool has_items = false;
    // Zelfde berekening als route_cost_for_mask, maar op basis van item_counts.
    // Dit wordt gebruikt tijdens moves waar orders tijdelijk verschuiven.
    for (const int item : items_by_snake_rank) {
        if (item_counts[static_cast<std::size_t>(item)] == 0U) {
            continue;
        }
        cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
        previous = item;
        has_items = true;
    }
    if (!has_items) {
        return 0.0;
    }
    cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance.end_index));
    return cost;
}

void update_search_batch_stats(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    SearchBatch& batch) {
    batch.cost = 0.0;
    batch.min_rank = 0;
    batch.max_rank = 0;

    int previous = instance.start_index;
    bool has_items = false;
    // Werk kost en snake-span tegelijk bij, zodat routes later op center
    // gesorteerd kunnen worden voor lokale neighbourhoods.
    for (const int item : items_by_snake_rank) {
        if (batch.item_counts[static_cast<std::size_t>(item)] == 0U) {
            continue;
        }
        if (!has_items) {
            batch.min_rank = instance.snake_rank.at(static_cast<std::size_t>(item));
            has_items = true;
        }
        batch.max_rank = instance.snake_rank.at(static_cast<std::size_t>(item));
        batch.cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
        previous = item;
    }
    if (has_items) {
        batch.cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance.end_index));
    }
}

double merged_route_cost(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::uint64_t>& left_mask,
    const std::vector<std::uint64_t>& right_mask) {
    double cost = 0.0;
    int previous = instance.start_index;
    bool has_items = false;
    // Bereken de kost van de union van twee routes zonder eerst een nieuwe
    // vector van items op te bouwen.
    for (const int item : items_by_snake_rank) {
        const std::size_t item_index = static_cast<std::size_t>(item);
        const std::uint64_t bit = std::uint64_t{1} << (item_index % 64U);
        const bool present = ((left_mask[item_index / 64U] | right_mask[item_index / 64U]) & bit) != 0U;
        if (!present) {
            continue;
        }
        cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(item));
        previous = item;
        has_items = true;
    }
    if (!has_items) {
        return 0.0;
    }
    cost += instance.distances.at(static_cast<std::size_t>(previous)).at(static_cast<std::size_t>(instance.end_index));
    return cost;
}

void merge_mask_into(std::vector<std::uint64_t>& target, const std::vector<std::uint64_t>& source) {
    for (std::size_t word = 0; word < target.size(); ++word) {
        target[word] |= source[word];
    }
}

void apply_order_count_delta(
    std::vector<std::uint16_t>& item_counts,
    const std::vector<int>& order_items,
    int delta) {
    for (const int item : order_items) {
        std::uint16_t& count = item_counts[static_cast<std::size_t>(item)];
        if (delta > 0) {
            ++count;
        } else {
            --count;
        }
    }
}

std::vector<std::uint16_t> build_item_counts(
    const ProblemInstance& instance,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::vector<int>& order_indices) {
    std::vector<std::uint16_t> counts(instance.item_ids.size(), 0U);
    for (const int order_index : order_indices) {
        apply_order_count_delta(counts, order_items_by_snake_rank[static_cast<std::size_t>(order_index)], 1);
    }
    return counts;
}

std::vector<SearchBatch> build_search_batches(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::vector<Batch>& batches) {
    std::vector<SearchBatch> search_batches;
    search_batches.reserve(batches.size());
    for (const Batch& batch : batches) {
        SearchBatch search_batch;
        search_batch.batch = batch;
        search_batch.item_counts = build_item_counts(instance, order_items_by_snake_rank, batch.order_indices);
        update_search_batch_stats(instance, items_by_snake_rank, search_batch);
        search_batches.push_back(std::move(search_batch));
    }
    return search_batches;
}

double span_center(const SearchBatch& batch) {
    return (static_cast<double>(batch.min_rank) + static_cast<double>(batch.max_rank)) / 2.0;
}

double simulated_swap_cost(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& left,
    SearchBatch& right,
    std::size_t left_pos,
    std::size_t right_pos) {
    const int left_order = left.batch.order_indices[left_pos];
    const int right_order = right.batch.order_indices[right_pos];
    const std::vector<int>& left_items = order_items_by_snake_rank[static_cast<std::size_t>(left_order)];
    const std::vector<int>& right_items = order_items_by_snake_rank[static_cast<std::size_t>(right_order)];

    apply_order_count_delta(left.item_counts, left_items, -1);
    apply_order_count_delta(left.item_counts, right_items, 1);
    apply_order_count_delta(right.item_counts, right_items, -1);
    apply_order_count_delta(right.item_counts, left_items, 1);
    const double cost = route_cost_for_counts(instance, items_by_snake_rank, left.item_counts) +
                        route_cost_for_counts(instance, items_by_snake_rank, right.item_counts);
    apply_order_count_delta(left.item_counts, right_items, -1);
    apply_order_count_delta(left.item_counts, left_items, 1);
    apply_order_count_delta(right.item_counts, left_items, -1);
    apply_order_count_delta(right.item_counts, right_items, 1);
    return cost;
}

double simulated_move_cost(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& source,
    SearchBatch& destination,
    std::size_t source_pos) {
    const int moved_order = source.batch.order_indices[source_pos];
    const std::vector<int>& moved_items = order_items_by_snake_rank[static_cast<std::size_t>(moved_order)];

    apply_order_count_delta(source.item_counts, moved_items, -1);
    apply_order_count_delta(destination.item_counts, moved_items, 1);
    const double cost = route_cost_for_counts(instance, items_by_snake_rank, source.item_counts) +
                        route_cost_for_counts(instance, items_by_snake_rank, destination.item_counts);
    apply_order_count_delta(source.item_counts, moved_items, 1);
    apply_order_count_delta(destination.item_counts, moved_items, -1);
    return cost;
}

LocalAction find_best_pair_action(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& left,
    SearchBatch& right) {
    LocalAction best;
    const double current_cost = left.cost + right.cost;

    // Test alle order-swaps tussen twee routes.
    for (std::size_t left_pos = 0; left_pos < left.batch.order_indices.size(); ++left_pos) {
        for (std::size_t right_pos = 0; right_pos < right.batch.order_indices.size(); ++right_pos) {
            const double candidate_cost = simulated_swap_cost(
                instance, items_by_snake_rank, order_items_by_snake_rank, left, right, left_pos, right_pos);
            const double delta = candidate_cost - current_cost;
            if (delta < best.delta - 1e-9) {
                best = {LocalActionType::swap_orders, left_pos, right_pos, delta};
            }
        }
    }

    if (!right.batch.order_indices.empty() &&
        right.batch.order_indices.size() < instance.nmax &&
        !left.batch.order_indices.empty()) {
        // Test relocaties van links naar rechts, alleen als de capaciteit dat toelaat.
        for (std::size_t left_pos = 0; left_pos < left.batch.order_indices.size(); ++left_pos) {
            const double candidate_cost = simulated_move_cost(
                instance, items_by_snake_rank, order_items_by_snake_rank, left, right, left_pos);
            const double delta = candidate_cost - current_cost;
            if (delta < best.delta - 1e-9) {
                best = {LocalActionType::move_left_to_right, left_pos, 0, delta};
            }
        }
    }

    if (!left.batch.order_indices.empty() &&
        left.batch.order_indices.size() < instance.nmax &&
        !right.batch.order_indices.empty()) {
        // Test relocaties van rechts naar links.
        for (std::size_t right_pos = 0; right_pos < right.batch.order_indices.size(); ++right_pos) {
            const double candidate_cost = simulated_move_cost(
                instance, items_by_snake_rank, order_items_by_snake_rank, right, left, right_pos);
            const double delta = candidate_cost - current_cost;
            if (delta < best.delta - 1e-9) {
                best = {LocalActionType::move_right_to_left, 0, right_pos, delta};
            }
        }
    }

    return best;
}

void apply_local_action(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& left,
    SearchBatch& right,
    const LocalAction& action) {
    // Pas exact dezelfde wijziging toe die eerder gesimuleerd werd en werk daarna
    // de routekosten opnieuw bij. De functie accepteert geen verslechteringen;
    // die filtering gebeurt in find_best_pair_action/improve_nearby_pairs.
    if (action.type == LocalActionType::swap_orders) {
        const int left_order = left.batch.order_indices[action.left_pos];
        const int right_order = right.batch.order_indices[action.right_pos];
        apply_order_count_delta(left.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(left_order)], -1);
        apply_order_count_delta(left.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(right_order)], 1);
        apply_order_count_delta(right.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(right_order)], -1);
        apply_order_count_delta(right.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(left_order)], 1);
        std::swap(left.batch.order_indices[action.left_pos], right.batch.order_indices[action.right_pos]);
    } else if (action.type == LocalActionType::move_left_to_right) {
        const int moved_order = left.batch.order_indices[action.left_pos];
        apply_order_count_delta(left.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(moved_order)], -1);
        apply_order_count_delta(right.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(moved_order)], 1);
        left.batch.order_indices.erase(left.batch.order_indices.begin() + static_cast<std::ptrdiff_t>(action.left_pos));
        right.batch.order_indices.push_back(moved_order);
    } else if (action.type == LocalActionType::move_right_to_left) {
        const int moved_order = right.batch.order_indices[action.right_pos];
        apply_order_count_delta(right.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(moved_order)], -1);
        apply_order_count_delta(left.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(moved_order)], 1);
        right.batch.order_indices.erase(right.batch.order_indices.begin() + static_cast<std::ptrdiff_t>(action.right_pos));
        left.batch.order_indices.push_back(moved_order);
    }

    update_search_batch_stats(instance, items_by_snake_rank, left);
    update_search_batch_stats(instance, items_by_snake_rank, right);
}

double order_rank_center_value(
    const ProblemInstance& instance,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    int order_index) {
    const std::vector<int>& items = order_items_by_snake_rank[static_cast<std::size_t>(order_index)];
    if (items.empty()) {
        return 0.0;
    }
    const int first_rank = instance.snake_rank.at(static_cast<std::size_t>(items.front()));
    const int last_rank = instance.snake_rank.at(static_cast<std::size_t>(items.back()));
    return (static_cast<double>(first_rank) + static_cast<double>(last_rank)) / 2.0;
}

int order_rank_span_value(
    const ProblemInstance& instance,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    int order_index) {
    const std::vector<int>& items = order_items_by_snake_rank[static_cast<std::size_t>(order_index)];
    if (items.empty()) {
        return 0;
    }
    const int first_rank = instance.snake_rank.at(static_cast<std::size_t>(items.front()));
    const int last_rank = instance.snake_rank.at(static_cast<std::size_t>(items.back()));
    return last_rank - first_rank;
}

SearchBatch make_search_batch(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::string& wave_id,
    std::vector<int> order_indices) {
    SearchBatch batch;
    batch.batch.wave_id = wave_id;
    batch.batch.order_indices = std::move(order_indices);
    batch.item_counts = build_item_counts(instance, order_items_by_snake_rank, batch.batch.order_indices);
    update_search_batch_stats(instance, items_by_snake_rank, batch);
    return batch;
}

double incremental_cost_for_order(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const SearchBatch& batch,
    int order_index) {
    std::vector<std::uint16_t> counts = batch.item_counts;
    apply_order_count_delta(counts, order_items_by_snake_rank[static_cast<std::size_t>(order_index)], 1);
    return route_cost_for_counts(instance, items_by_snake_rank, counts) - batch.cost;
}

void add_order_to_search_batch(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& batch,
    int order_index) {
    batch.batch.order_indices.push_back(order_index);
    apply_order_count_delta(batch.item_counts, order_items_by_snake_rank[static_cast<std::size_t>(order_index)], 1);
    update_search_batch_stats(instance, items_by_snake_rank, batch);
}

void polish_rebuilt_pair(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& left,
    SearchBatch& right) {
    // Na een volledige rebuild kunnen er nog kleine pairwise verbeteringen openstaan.
    // Deze polish voert de beste swap/relocate uit zolang de pair-kost daalt.
    while (!left.batch.order_indices.empty() && !right.batch.order_indices.empty()) {
        const LocalAction action =
            find_best_pair_action(instance, items_by_snake_rank, order_items_by_snake_rank, left, right);
        if (action.type == LocalActionType::none) {
            break;
        }
        apply_local_action(instance, items_by_snake_rank, order_items_by_snake_rank, left, right, action);
    }
}

void consider_rebuilt_partition(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::string& wave_id,
    const std::vector<int>& left_orders,
    const std::vector<int>& right_orders,
    double current_cost,
    double& best_cost,
    SearchBatch& best_left,
    SearchBatch& best_right) {
    // Evalueer een kandidaatverdeling van dezelfde orders over twee routes.
    // Alleen haalbare verdelingen met een strikt lagere kost vervangen de beste kandidaat.
    if (left_orders.empty() || right_orders.empty()) {
        return;
    }
    if (left_orders.size() > instance.nmax || right_orders.size() > instance.nmax) {
        return;
    }

    SearchBatch candidate_left =
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, left_orders);
    SearchBatch candidate_right =
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, right_orders);
    polish_rebuilt_pair(instance, items_by_snake_rank, order_items_by_snake_rank, candidate_left, candidate_right);
    if (candidate_left.batch.order_indices.empty() || candidate_right.batch.order_indices.empty()) {
        return;
    }
    if (candidate_left.batch.order_indices.size() > instance.nmax ||
        candidate_right.batch.order_indices.size() > instance.nmax) {
        return;
    }

    const double candidate_cost = candidate_left.cost + candidate_right.cost;
    if (candidate_cost < current_cost - 1e-9 && candidate_cost < best_cost - 1e-9) {
        best_cost = candidate_cost;
        best_left = std::move(candidate_left);
        best_right = std::move(candidate_right);
    }
}

void consider_sorted_split(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::string& wave_id,
    std::vector<int> combined_orders,
    double current_cost,
    double& best_cost,
    SearchBatch& best_left,
    SearchBatch& best_right) {
    // Snelle deterministische kandidaat: de orders zijn al gesorteerd door de caller
    // en worden hier in twee haalbare routes geknipt.
    const std::size_t target_left = std::min<std::size_t>(
        instance.nmax,
        std::max<std::size_t>(1U, combined_orders.size() / 2U));
    std::vector<int> left_orders(
        combined_orders.begin(),
        combined_orders.begin() + static_cast<std::ptrdiff_t>(target_left));
    std::vector<int> right_orders(
        combined_orders.begin() + static_cast<std::ptrdiff_t>(target_left),
        combined_orders.end());
    consider_rebuilt_partition(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        wave_id,
        left_orders,
        right_orders,
        current_cost,
        best_cost,
        best_left,
        best_right);
}

void consider_greedy_seed_rebuild(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::string& wave_id,
    std::vector<int> combined_orders,
    double current_cost,
    double& best_cost,
    SearchBatch& best_left,
    SearchBatch& best_right) {
    // Bouw twee routes opnieuw op vanuit twee seed-orders aan de uitersten van de
    // snake-center. Overige orders gaan telkens naar de route met de laagste extra kost.
    if (combined_orders.size() <= 1U) {
        return;
    }

    const auto minmax = std::minmax_element(
        combined_orders.begin(),
        combined_orders.end(),
        [&](const int left, const int right) {
            return order_rank_center_value(instance, order_items_by_snake_rank, left) <
                   order_rank_center_value(instance, order_items_by_snake_rank, right);
        });
    if (minmax.first == combined_orders.end() || minmax.second == combined_orders.end() ||
        *minmax.first == *minmax.second) {
        return;
    }

    const int left_seed = *minmax.first;
    const int right_seed = *minmax.second;
    SearchBatch left = make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, {left_seed});
    SearchBatch right =
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, {right_seed});

    combined_orders.erase(std::remove(combined_orders.begin(), combined_orders.end(), left_seed), combined_orders.end());
    combined_orders.erase(std::remove(combined_orders.begin(), combined_orders.end(), right_seed), combined_orders.end());
    std::sort(combined_orders.begin(), combined_orders.end(), [&](const int left_order, const int right_order) {
        const int left_span = order_rank_span_value(instance, order_items_by_snake_rank, left_order);
        const int right_span = order_rank_span_value(instance, order_items_by_snake_rank, right_order);
        if (left_span != right_span) {
            return left_span > right_span;
        }
        const std::size_t left_items = order_items_by_snake_rank[static_cast<std::size_t>(left_order)].size();
        const std::size_t right_items = order_items_by_snake_rank[static_cast<std::size_t>(right_order)].size();
        if (left_items != right_items) {
            return left_items > right_items;
        }
        return left_order < right_order;
    });

    for (const int order_index : combined_orders) {
        if (left.batch.order_indices.size() >= instance.nmax) {
            add_order_to_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, right, order_index);
            continue;
        }
        if (right.batch.order_indices.size() >= instance.nmax) {
            add_order_to_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, left, order_index);
            continue;
        }

        const double left_increment =
            incremental_cost_for_order(instance, items_by_snake_rank, order_items_by_snake_rank, left, order_index);
        const double right_increment =
            incremental_cost_for_order(instance, items_by_snake_rank, order_items_by_snake_rank, right, order_index);
        if (left_increment < right_increment - 1e-9 ||
            (std::abs(left_increment - right_increment) <= 1e-9 &&
             left.batch.order_indices.size() <= right.batch.order_indices.size())) {
            add_order_to_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, left, order_index);
        } else {
            add_order_to_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, right, order_index);
        }
    }

    consider_rebuilt_partition(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        wave_id,
        left.batch.order_indices,
        right.batch.order_indices,
        current_cost,
        best_cost,
        best_left,
        best_right);
}

void consider_regret_seed_rebuild(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::string& wave_id,
    std::vector<int> combined_orders,
    double current_cost,
    double& best_cost,
    SearchBatch& best_left,
    SearchBatch& best_right) {
    // Regret-greedy plaatst eerst orders waarvoor het verschil tussen de beste
    // en tweede beste route groot is. Zo blijven moeilijke orders niet tot het einde.
    if (combined_orders.size() <= 1U) {
        return;
    }

    const auto minmax = std::minmax_element(
        combined_orders.begin(),
        combined_orders.end(),
        [&](const int left, const int right) {
            return order_rank_center_value(instance, order_items_by_snake_rank, left) <
                   order_rank_center_value(instance, order_items_by_snake_rank, right);
        });
    if (minmax.first == combined_orders.end() || minmax.second == combined_orders.end() ||
        *minmax.first == *minmax.second) {
        return;
    }

    const int left_seed = *minmax.first;
    const int right_seed = *minmax.second;
    SearchBatch left = make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, {left_seed});
    SearchBatch right =
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, {right_seed});

    combined_orders.erase(std::remove(combined_orders.begin(), combined_orders.end(), left_seed), combined_orders.end());
    combined_orders.erase(std::remove(combined_orders.begin(), combined_orders.end(), right_seed), combined_orders.end());

    while (!combined_orders.empty()) {
        std::size_t best_pos = 0;
        bool best_to_left = true;
        double best_regret = -std::numeric_limits<double>::infinity();
        double best_increment = std::numeric_limits<double>::infinity();
        int best_span = -1;
        std::size_t best_item_count = 0;

        for (std::size_t pos = 0; pos < combined_orders.size(); ++pos) {
            const int order_index = combined_orders[pos];
            const bool left_has_capacity = left.batch.order_indices.size() < instance.nmax;
            const bool right_has_capacity = right.batch.order_indices.size() < instance.nmax;
            if (!left_has_capacity && !right_has_capacity) {
                return;
            }

            double left_increment = std::numeric_limits<double>::infinity();
            double right_increment = std::numeric_limits<double>::infinity();
            if (left_has_capacity) {
                left_increment =
                    incremental_cost_for_order(instance, items_by_snake_rank, order_items_by_snake_rank, left, order_index);
            }
            if (right_has_capacity) {
                right_increment =
                    incremental_cost_for_order(instance, items_by_snake_rank, order_items_by_snake_rank, right, order_index);
            }

            const bool to_left = left_increment <= right_increment;
            const double best_candidate_increment = std::min(left_increment, right_increment);
            const double worst_candidate_increment = std::max(left_increment, right_increment);
            const double regret = worst_candidate_increment - best_candidate_increment;
            const int span = order_rank_span_value(instance, order_items_by_snake_rank, order_index);
            const std::size_t item_count = order_items_by_snake_rank[static_cast<std::size_t>(order_index)].size();

            if (regret > best_regret + 1e-9 ||
                (std::abs(regret - best_regret) <= 1e-9 &&
                 (best_candidate_increment < best_increment - 1e-9 ||
                  (std::abs(best_candidate_increment - best_increment) <= 1e-9 &&
                   (span > best_span ||
                    (span == best_span &&
                     (item_count > best_item_count ||
                      (item_count == best_item_count && order_index < combined_orders[best_pos])))))))) {
                best_pos = pos;
                best_to_left = to_left;
                best_regret = regret;
                best_increment = best_candidate_increment;
                best_span = span;
                best_item_count = item_count;
            }
        }

        const int order_index = combined_orders[best_pos];
        if (best_to_left && left.batch.order_indices.size() < instance.nmax) {
            add_order_to_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, left, order_index);
        } else {
            add_order_to_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, right, order_index);
        }
        combined_orders.erase(combined_orders.begin() + static_cast<std::ptrdiff_t>(best_pos));
    }

    consider_rebuilt_partition(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        wave_id,
        left.batch.order_indices,
        right.batch.order_indices,
        current_cost,
        best_cost,
        best_left,
        best_right);
}

bool rebuild_pair_if_better(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& left,
    SearchBatch& right) {
    // Sterkere neighbourhood dan een swap: combineer twee routes en verdeel
    // alle orders opnieuw. We accepteren alleen als de totale kost daalt.
    if (left.batch.wave_id != right.batch.wave_id) {
        return false;
    }

    std::vector<int> combined_orders = left.batch.order_indices;
    combined_orders.insert(combined_orders.end(), right.batch.order_indices.begin(), right.batch.order_indices.end());
    if (combined_orders.size() <= instance.nmax || combined_orders.size() > 2U * instance.nmax) {
        return false;
    }

    const double current_cost = left.cost + right.cost;
    double best_cost = current_cost;
    SearchBatch best_left;
    SearchBatch best_right;

    std::vector<int> by_center = combined_orders;
    std::sort(by_center.begin(), by_center.end(), [&](const int left_order, const int right_order) {
        const double left_center = order_rank_center_value(instance, order_items_by_snake_rank, left_order);
        const double right_center = order_rank_center_value(instance, order_items_by_snake_rank, right_order);
        if (std::abs(left_center - right_center) > 1e-9) {
            return left_center < right_center;
        }
        return left_order < right_order;
    });
    consider_sorted_split(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        left.batch.wave_id,
        by_center,
        current_cost,
        best_cost,
        best_left,
        best_right);

    std::vector<int> by_span = combined_orders;
    std::sort(by_span.begin(), by_span.end(), [&](const int left_order, const int right_order) {
        const int left_span = order_rank_span_value(instance, order_items_by_snake_rank, left_order);
        const int right_span = order_rank_span_value(instance, order_items_by_snake_rank, right_order);
        if (left_span != right_span) {
            return left_span > right_span;
        }
        return left_order < right_order;
    });
    consider_sorted_split(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        left.batch.wave_id,
        by_span,
        current_cost,
        best_cost,
        best_left,
        best_right);

    consider_greedy_seed_rebuild(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        left.batch.wave_id,
        combined_orders,
        current_cost,
        best_cost,
        best_left,
        best_right);

    consider_regret_seed_rebuild(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        left.batch.wave_id,
        combined_orders,
        current_cost,
        best_cost,
        best_left,
        best_right);

    if (best_cost >= current_cost - 1e-9) {
        return false;
    }

    left = std::move(best_left);
    right = std::move(best_right);
    return true;
}

double total_rebuild_cost(const std::array<SearchBatch, 3>& routes) {
    return routes[0].cost + routes[1].cost + routes[2].cost;
}

std::array<std::size_t, 3> balanced_three_sizes(std::size_t order_count, std::size_t nmax) {
    std::array<std::size_t, 3> sizes{
        order_count / 3U,
        order_count / 3U,
        order_count / 3U,
    };
    for (std::size_t index = 0; index < order_count % 3U; ++index) {
        ++sizes[index];
    }

    for (std::size_t index = 0; index < sizes.size(); ++index) {
        if (sizes[index] == 0U) {
            sizes[index] = 1U;
        }
        if (sizes[index] > nmax) {
            sizes[index] = nmax;
        }
    }
    return sizes;
}

void polish_rebuilt_triple(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    std::array<SearchBatch, 3>& routes) {
    // Na een triple-rebuild kunnen nog kleine pairwise verbeteringen openstaan.
    // Daarom polijsten we de drie routeparen tot geen enkel paar nog verbetert.
    bool improved = true;
    while (improved) {
        improved = false;
        for (std::size_t left_index = 0; left_index < routes.size(); ++left_index) {
            for (std::size_t right_index = left_index + 1U; right_index < routes.size(); ++right_index) {
                while (!routes[left_index].batch.order_indices.empty() &&
                       !routes[right_index].batch.order_indices.empty()) {
                    const LocalAction action = find_best_pair_action(
                        instance,
                        items_by_snake_rank,
                        order_items_by_snake_rank,
                        routes[left_index],
                        routes[right_index]);
                    if (action.type == LocalActionType::none) {
                        break;
                    }
                    apply_local_action(
                        instance,
                        items_by_snake_rank,
                        order_items_by_snake_rank,
                        routes[left_index],
                        routes[right_index],
                        action);
                    improved = true;
                }
            }
        }
    }
}

void consider_rebuilt_triple_partition(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::string& wave_id,
    const std::array<std::vector<int>, 3>& order_groups,
    double current_cost,
    double& best_cost,
    std::array<SearchBatch, 3>& best_routes) {
    // Evalueer een volledige verdeling over drie routes.
    // Elke route moet niet-leeg blijven en binnen de capaciteit vallen.
    for (const std::vector<int>& orders : order_groups) {
        if (orders.empty() || orders.size() > instance.nmax) {
            return;
        }
    }

    std::array<SearchBatch, 3> candidate_routes{
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, order_groups[0]),
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, order_groups[1]),
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, order_groups[2]),
    };
    polish_rebuilt_triple(instance, items_by_snake_rank, order_items_by_snake_rank, candidate_routes);

    for (const SearchBatch& route : candidate_routes) {
        if (route.batch.order_indices.empty() || route.batch.order_indices.size() > instance.nmax) {
            return;
        }
    }

    const double candidate_cost = total_rebuild_cost(candidate_routes);
    if (candidate_cost < current_cost - 1e-9 && candidate_cost < best_cost - 1e-9) {
        best_cost = candidate_cost;
        best_routes = std::move(candidate_routes);
    }
}

void consider_sorted_triple_split(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::string& wave_id,
    const std::vector<int>& combined_orders,
    double current_cost,
    double& best_cost,
    std::array<SearchBatch, 3>& best_routes) {
    // Goedkope triple-kandidaat: sorteer vooraf op center/span en split in drie
    // ongeveer gebalanceerde groepen.
    const std::array<std::size_t, 3> sizes = balanced_three_sizes(combined_orders.size(), instance.nmax);
    if (sizes[0] + sizes[1] + sizes[2] != combined_orders.size()) {
        return;
    }

    std::array<std::vector<int>, 3> groups;
    std::size_t offset = 0;
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        groups[group_index].assign(
            combined_orders.begin() + static_cast<std::ptrdiff_t>(offset),
            combined_orders.begin() + static_cast<std::ptrdiff_t>(offset + sizes[group_index]));
        offset += sizes[group_index];
    }

    consider_rebuilt_triple_partition(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        wave_id,
        groups,
        current_cost,
        best_cost,
        best_routes);
}

std::array<int, 3> select_three_center_seeds(
    const ProblemInstance& instance,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    std::vector<int> combined_orders) {
    std::sort(combined_orders.begin(), combined_orders.end(), [&](const int left, const int right) {
        const double left_center = order_rank_center_value(instance, order_items_by_snake_rank, left);
        const double right_center = order_rank_center_value(instance, order_items_by_snake_rank, right);
        if (std::abs(left_center - right_center) > 1e-9) {
            return left_center < right_center;
        }
        return left < right;
    });
    return {
        combined_orders.front(),
        combined_orders[combined_orders.size() / 2U],
        combined_orders.back(),
    };
}

void remove_seed_orders(std::vector<int>& orders, const std::array<int, 3>& seeds) {
    for (const int seed : seeds) {
        const auto found = std::find(orders.begin(), orders.end(), seed);
        if (found != orders.end()) {
            orders.erase(found);
        }
    }
}

void consider_three_seed_rebuild(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::string& wave_id,
    std::vector<int> combined_orders,
    bool regret_ordering,
    double current_cost,
    double& best_cost,
    std::array<SearchBatch, 3>& best_routes) {
    // Drie-route variant van de seed rebuild. Met regret_ordering=true wordt
    // telkens de meest kritieke order eerst geplaatst; anders volgt een vaste volgorde.
    if (combined_orders.size() < 3U || combined_orders.size() > 3U * instance.nmax) {
        return;
    }

    const std::array<int, 3> seeds =
        select_three_center_seeds(instance, order_items_by_snake_rank, combined_orders);
    if (seeds[0] == seeds[1] || seeds[0] == seeds[2] || seeds[1] == seeds[2]) {
        return;
    }

    std::array<SearchBatch, 3> routes{
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, {seeds[0]}),
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, {seeds[1]}),
        make_search_batch(instance, items_by_snake_rank, order_items_by_snake_rank, wave_id, {seeds[2]}),
    };

    remove_seed_orders(combined_orders, seeds);
    if (!regret_ordering) {
        std::sort(combined_orders.begin(), combined_orders.end(), [&](const int left_order, const int right_order) {
            const int left_span = order_rank_span_value(instance, order_items_by_snake_rank, left_order);
            const int right_span = order_rank_span_value(instance, order_items_by_snake_rank, right_order);
            if (left_span != right_span) {
                return left_span > right_span;
            }
            return left_order < right_order;
        });
    }

    while (!combined_orders.empty()) {
        std::size_t selected_pos = 0;
        std::size_t selected_route = 0;
        double best_regret = -std::numeric_limits<double>::infinity();
        double best_increment = std::numeric_limits<double>::infinity();
        int best_span = -1;
        std::size_t best_item_count = 0;

        const std::size_t candidate_limit = regret_ordering ? combined_orders.size() : 1U;
        for (std::size_t pos = 0; pos < candidate_limit; ++pos) {
            const int order_index = combined_orders[pos];
            std::array<double, 3> increments{
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
            };

            for (std::size_t route_index = 0; route_index < routes.size(); ++route_index) {
                if (routes[route_index].batch.order_indices.size() < instance.nmax) {
                    increments[route_index] = incremental_cost_for_order(
                        instance,
                        items_by_snake_rank,
                        order_items_by_snake_rank,
                        routes[route_index],
                        order_index);
                }
            }

            std::array<std::size_t, 3> route_order{0U, 1U, 2U};
            std::sort(route_order.begin(), route_order.end(), [&](const std::size_t left, const std::size_t right) {
                if (std::abs(increments[left] - increments[right]) > 1e-9) {
                    return increments[left] < increments[right];
                }
                return routes[left].batch.order_indices.size() < routes[right].batch.order_indices.size();
            });
            if (!std::isfinite(increments[route_order[0]])) {
                return;
            }

            const double regret = std::isfinite(increments[route_order[1]])
                                      ? increments[route_order[1]] - increments[route_order[0]]
                                      : std::numeric_limits<double>::infinity();
            const int span = order_rank_span_value(instance, order_items_by_snake_rank, order_index);
            const std::size_t item_count = order_items_by_snake_rank[static_cast<std::size_t>(order_index)].size();
            if (!regret_ordering ||
                regret > best_regret + 1e-9 ||
                (std::abs(regret - best_regret) <= 1e-9 &&
                 (increments[route_order[0]] < best_increment - 1e-9 ||
                  (std::abs(increments[route_order[0]] - best_increment) <= 1e-9 &&
                   (span > best_span ||
                    (span == best_span &&
                     (item_count > best_item_count ||
                      (item_count == best_item_count && order_index < combined_orders[selected_pos])))))))) {
                selected_pos = pos;
                selected_route = route_order[0];
                best_regret = regret;
                best_increment = increments[route_order[0]];
                best_span = span;
                best_item_count = item_count;
            }
        }

        const int order_index = combined_orders[selected_pos];
        add_order_to_search_batch(
            instance,
            items_by_snake_rank,
            order_items_by_snake_rank,
            routes[selected_route],
            order_index);
        combined_orders.erase(combined_orders.begin() + static_cast<std::ptrdiff_t>(selected_pos));
    }

    std::array<std::vector<int>, 3> groups{
        routes[0].batch.order_indices,
        routes[1].batch.order_indices,
        routes[2].batch.order_indices,
    };
    consider_rebuilt_triple_partition(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        wave_id,
        groups,
        current_cost,
        best_cost,
        best_routes);
}

bool rebuild_triple_if_better(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    SearchBatch& first,
    SearchBatch& second,
    SearchBatch& third) {
    // Triple rebuild kan verbeteringen vinden die geen enkel routepaar afzonderlijk
    // kan bereiken. Dit was een van de belangrijkste late scoreverbeteringen.
    if (first.batch.wave_id != second.batch.wave_id || first.batch.wave_id != third.batch.wave_id) {
        return false;
    }

    std::vector<int> combined_orders = first.batch.order_indices;
    combined_orders.insert(combined_orders.end(), second.batch.order_indices.begin(), second.batch.order_indices.end());
    combined_orders.insert(combined_orders.end(), third.batch.order_indices.begin(), third.batch.order_indices.end());
    if (combined_orders.size() < 3U || combined_orders.size() > 3U * instance.nmax) {
        return false;
    }

    const double current_cost = first.cost + second.cost + third.cost;
    double best_cost = current_cost;
    std::array<SearchBatch, 3> best_routes;

    std::vector<int> by_center = combined_orders;
    std::sort(by_center.begin(), by_center.end(), [&](const int left_order, const int right_order) {
        const double left_center = order_rank_center_value(instance, order_items_by_snake_rank, left_order);
        const double right_center = order_rank_center_value(instance, order_items_by_snake_rank, right_order);
        if (std::abs(left_center - right_center) > 1e-9) {
            return left_center < right_center;
        }
        return left_order < right_order;
    });
    consider_sorted_triple_split(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        first.batch.wave_id,
        by_center,
        current_cost,
        best_cost,
        best_routes);

    consider_three_seed_rebuild(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        first.batch.wave_id,
        combined_orders,
        false,
        current_cost,
        best_cost,
        best_routes);
    consider_three_seed_rebuild(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        first.batch.wave_id,
        combined_orders,
        true,
        current_cost,
        best_cost,
        best_routes);

    if (best_cost >= current_cost - 1e-9) {
        return false;
    }

    first = std::move(best_routes[0]);
    second = std::move(best_routes[1]);
    third = std::move(best_routes[2]);
    return true;
}

void remove_empty_search_batches(std::vector<SearchBatch>& batches) {
    // Na relocaties of route-eliminatie kunnen routes leeg worden.
    // Lege routes worden niet naar de submission geschreven.
    batches.erase(
        std::remove_if(batches.begin(), batches.end(), [](const SearchBatch& batch) {
            return batch.batch.order_indices.empty();
        }),
        batches.end());
}

double simulated_add_cost(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    TrialDestination& destination,
    int order_index) {
    // Simuleer de kost van een order toevoegen zonder de destination definitief te wijzigen.
    const std::vector<int>& items = order_items_by_snake_rank[static_cast<std::size_t>(order_index)];
    apply_order_count_delta(destination.item_counts, items, 1);
    const double new_cost = route_cost_for_counts(instance, items_by_snake_rank, destination.item_counts);
    apply_order_count_delta(destination.item_counts, items, -1);
    return new_cost;
}

std::vector<DestinationChoice> ranked_destination_choices(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    EliminationState& state,
    int order_index,
    int max_choices) {
    // Sorteer mogelijke bestemmingen voor een order op incrementele kost.
    // max_choices beperkt de branching factor van de beam search.
    std::vector<DestinationChoice> choices;
    for (std::size_t destination_index = 0; destination_index < state.destinations.size(); ++destination_index) {
        TrialDestination& destination = state.destinations[destination_index];
        if (destination.order_count >= instance.nmax) {
            continue;
        }

        const double new_cost =
            simulated_add_cost(instance, items_by_snake_rank, order_items_by_snake_rank, destination, order_index);
        choices.push_back({destination_index, new_cost - destination.cost, new_cost});
    }

    std::sort(choices.begin(), choices.end(), [](const DestinationChoice& left, const DestinationChoice& right) {
        if (std::abs(left.incremental_cost - right.incremental_cost) > 1e-9) {
            return left.incremental_cost < right.incremental_cost;
        }
        return left.destination_index < right.destination_index;
    });
    if (max_choices > 0 && choices.size() > static_cast<std::size_t>(max_choices)) {
        choices.resize(static_cast<std::size_t>(max_choices));
    }
    return choices;
}

void assign_order_to_destination(
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    EliminationState& state,
    int order_index,
    const DestinationChoice& choice) {
    // Pas een gekozen beam-stap toe op een gekopieerde eliminatiestate.
    TrialDestination& destination = state.destinations[choice.destination_index];
    apply_order_count_delta(
        destination.item_counts,
        order_items_by_snake_rank[static_cast<std::size_t>(order_index)],
        1);
    ++destination.order_count;
    destination.cost = choice.new_cost;
    destination.assigned_orders.push_back(order_index);
    state.incremental_cost += choice.incremental_cost;
}

int order_span_width(
    const ProblemInstance& instance,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    int order_index) {
    const std::vector<int>& items = order_items_by_snake_rank[static_cast<std::size_t>(order_index)];
    if (items.empty()) {
        return 0;
    }
    const int first_rank = instance.snake_rank.at(static_cast<std::size_t>(items.front()));
    const int last_rank = instance.snake_rank.at(static_cast<std::size_t>(items.back()));
    return last_rank - first_rank;
}

double initial_destination_regret(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::vector<TrialDestination>& destinations,
    int order_index) {
    // Regret voor de startsortering van route-eliminatie:
    // orders met weinig goede bestemmingen worden vroeger verwerkt.
    double best = 0.0;
    double second_best = 0.0;
    bool has_best = false;
    bool has_second = false;
    for (TrialDestination destination : destinations) {
        if (destination.order_count >= instance.nmax) {
            continue;
        }
        const double new_cost =
            simulated_add_cost(instance, items_by_snake_rank, order_items_by_snake_rank, destination, order_index);
        const double incremental_cost = new_cost - destination.cost;
        if (!has_best || incremental_cost < best) {
            second_best = best;
            has_second = has_best;
            best = incremental_cost;
            has_best = true;
        } else if (!has_second || incremental_cost < second_best) {
            second_best = incremental_cost;
            has_second = true;
        }
    }
    if (!has_best) {
        return -std::numeric_limits<double>::infinity();
    }
    if (!has_second) {
        return std::numeric_limits<double>::infinity();
    }
    return second_best - best;
}

void add_unique_ordering(std::vector<std::vector<int>>& orderings, std::vector<int> ordering) {
    for (const std::vector<int>& existing : orderings) {
        if (existing == ordering) {
            return;
        }
    }
    orderings.push_back(std::move(ordering));
}

std::vector<std::vector<int>> build_source_orderings(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::vector<TrialDestination>& destinations,
    const std::vector<int>& source_orders,
    int ordering_trials) {
    // Route-eliminatie is gevoelig voor de ordervolgorde waarin orders worden geplaatst.
    // Daarom testen we meerdere deterministische volgordes en kiezen we de beste.
    std::vector<std::vector<int>> orderings;
    std::vector<int> ordering = source_orders;

    std::sort(ordering.begin(), ordering.end(), [&](const int left, const int right) {
        return order_items_by_snake_rank[static_cast<std::size_t>(left)].size() >
               order_items_by_snake_rank[static_cast<std::size_t>(right)].size();
    });
    add_unique_ordering(orderings, ordering);

    if (ordering_trials >= 2) {
        ordering = source_orders;
        std::sort(ordering.begin(), ordering.end(), [&](const int left, const int right) {
            return order_items_by_snake_rank[static_cast<std::size_t>(left)].size() <
                   order_items_by_snake_rank[static_cast<std::size_t>(right)].size();
        });
        add_unique_ordering(orderings, ordering);
    }

    if (ordering_trials >= 3) {
        ordering = source_orders;
        std::sort(ordering.begin(), ordering.end(), [&](const int left, const int right) {
            return order_span_width(instance, order_items_by_snake_rank, left) >
                   order_span_width(instance, order_items_by_snake_rank, right);
        });
        add_unique_ordering(orderings, ordering);
    }

    if (ordering_trials >= 4) {
        ordering = source_orders;
        std::sort(ordering.begin(), ordering.end(), [&](const int left, const int right) {
            const double left_regret = initial_destination_regret(
                instance, items_by_snake_rank, order_items_by_snake_rank, destinations, left);
            const double right_regret = initial_destination_regret(
                instance, items_by_snake_rank, order_items_by_snake_rank, destinations, right);
            if (std::abs(left_regret - right_regret) > 1e-9) {
                return left_regret > right_regret;
            }
            return left < right;
        });
        add_unique_ordering(orderings, ordering);
    }

    return orderings;
}

bool run_elimination_beam_for_ordering(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const SearchBatch& source,
    const std::vector<TrialDestination>& destinations,
    const std::vector<int>& source_orders,
    int beam_width,
    int candidates_per_order,
    EliminationState& best_state) {
    // Beam search probeert alle orders uit een bronroute in andere routes te plaatsen.
    // Alleen gedeeltelijke oplossingen die nog goedkoper kunnen zijn dan de bronroute
    // worden bijgehouden.
    std::vector<EliminationState> states;
    states.push_back({destinations, 0.0});
    for (const int order_index : source_orders) {
        std::vector<EliminationState> next_states;
        const std::size_t reserve_count =
            states.size() * static_cast<std::size_t>(std::max(1, candidates_per_order));
        next_states.reserve(reserve_count);

        for (EliminationState& state : states) {
            std::vector<DestinationChoice> choices = ranked_destination_choices(
                instance,
                items_by_snake_rank,
                order_items_by_snake_rank,
                state,
                order_index,
                candidates_per_order);
            for (const DestinationChoice& choice : choices) {
                if (state.incremental_cost + choice.incremental_cost >= source.cost - 1e-9) {
                    continue;
                }
                EliminationState next_state = state;
                assign_order_to_destination(order_items_by_snake_rank, next_state, order_index, choice);
                next_states.push_back(std::move(next_state));
            }
        }

        if (next_states.empty()) {
            return false;
        }

        std::sort(next_states.begin(), next_states.end(), [](const EliminationState& left, const EliminationState& right) {
            return left.incremental_cost < right.incremental_cost;
        });
        const int width = std::max(1, beam_width);
        if (next_states.size() > static_cast<std::size_t>(width)) {
            next_states.resize(static_cast<std::size_t>(width));
        }
        states = std::move(next_states);
    }

    best_state = std::move(states.front());
    return true;
}

bool build_elimination_plan(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    const std::vector<SearchBatch>& batches,
    std::size_t source_index,
    int max_source_batch_size,
    int beam_width,
    int candidates_per_order,
    int ordering_trials,
    EliminationPlan& plan) {
    // Bouw het beste plan om een specifieke bronroute te verwijderen.
    // Als de vrije capaciteit onvoldoende is of de beste herverdeling duurder is,
    // wordt de route niet verwijderd.
    const SearchBatch& source = batches[source_index];
    if (source.batch.order_indices.empty() ||
        source.batch.order_indices.size() > static_cast<std::size_t>(max_source_batch_size)) {
        return false;
    }

    std::vector<TrialDestination> destinations;
    std::size_t free_capacity = 0;
    for (std::size_t batch_index = 0; batch_index < batches.size(); ++batch_index) {
        if (batch_index == source_index || batches[batch_index].batch.order_indices.empty()) {
            continue;
        }
        if (batches[batch_index].batch.order_indices.size() >= instance.nmax) {
            continue;
        }

        TrialDestination destination;
        destination.batch_index = batch_index;
        destination.item_counts = batches[batch_index].item_counts;
        destination.order_count = batches[batch_index].batch.order_indices.size();
        destination.cost = batches[batch_index].cost;
        free_capacity += instance.nmax - destination.order_count;
        destinations.push_back(std::move(destination));
    }

    if (free_capacity < source.batch.order_indices.size() || destinations.empty()) {
        return false;
    }

    const std::vector<std::vector<int>> source_orderings = build_source_orderings(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        destinations,
        source.batch.order_indices,
        ordering_trials);

    EliminationState best_state;
    bool found_state = false;
    for (const std::vector<int>& source_orders : source_orderings) {
        EliminationState candidate_state;
        if (!run_elimination_beam_for_ordering(
                instance,
                items_by_snake_rank,
                order_items_by_snake_rank,
                source,
                destinations,
                source_orders,
                beam_width,
                candidates_per_order,
                candidate_state)) {
            continue;
        }
        if (!found_state || candidate_state.incremental_cost < best_state.incremental_cost) {
            best_state = std::move(candidate_state);
            found_state = true;
        }
    }

    if (!found_state) {
        return false;
    }
    if (best_state.incremental_cost >= source.cost - 1e-9) {
        return false;
    }

    plan.source_index = source_index;
    plan.destinations = best_state.destinations;
    plan.saving = source.cost - best_state.incremental_cost;
    return true;
}

void apply_elimination_plan(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    std::vector<SearchBatch>& batches,
    const EliminationPlan& plan) {
    // Schrijf de gekozen herverdeling terug naar de echte search_batches.
    for (const TrialDestination& destination : plan.destinations) {
        if (destination.assigned_orders.empty()) {
            continue;
        }
        SearchBatch& target = batches[destination.batch_index];
        for (const int order_index : destination.assigned_orders) {
            target.batch.order_indices.push_back(order_index);
        }
        target.item_counts = destination.item_counts;
        update_search_batch_stats(instance, items_by_snake_rank, target);
    }

    SearchBatch& source = batches[plan.source_index];
    source.batch.order_indices.clear();
    std::fill(source.item_counts.begin(), source.item_counts.end(), 0U);
    update_search_batch_stats(instance, items_by_snake_rank, source);
}

std::size_t eliminate_routes(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    std::vector<SearchBatch>& batches,
    int passes,
    int max_source_batch_size,
    int beam_width,
    int candidates_per_order,
    int ordering_trials) {
    // Zoek per pass het beste route-eliminatieplan. Kleine routes worden eerst
    // bekeken omdat die het vaakst volledig in vrije capaciteit passen.
    if (passes <= 0 || max_source_batch_size <= 0) {
        return 0;
    }

    std::size_t eliminated_count = 0;
    for (int pass = 0; pass < passes; ++pass) {
        EliminationPlan best_plan;
        bool found_plan = false;
        std::vector<std::size_t> sorted_indices(batches.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](const std::size_t left, const std::size_t right) {
            return batches[left].batch.order_indices.size() < batches[right].batch.order_indices.size();
        });

        for (const std::size_t source_index : sorted_indices) {
            if (source_index >= batches.size() || batches[source_index].batch.order_indices.empty()) {
                continue;
            }
            EliminationPlan plan;
            if (!build_elimination_plan(
                    instance,
                    items_by_snake_rank,
                    order_items_by_snake_rank,
                    batches,
                    source_index,
                    max_source_batch_size,
                    beam_width,
                    candidates_per_order,
                    ordering_trials,
                    plan)) {
                continue;
            }
            if (!found_plan || plan.saving > best_plan.saving + 1e-9) {
                best_plan = std::move(plan);
                found_plan = true;
            }
        }

        if (!found_plan) {
            break;
        }

        apply_elimination_plan(instance, items_by_snake_rank, batches, best_plan);
        ++eliminated_count;
        remove_empty_search_batches(batches);
    }

    return eliminated_count;
}

std::size_t improve_nearby_pairs(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    std::vector<SearchBatch>& search_batches,
    int passes,
    int neighbor_window) {
    // Eenvoudige local search: sorteer routes op snake-center en probeer alleen
    // nabije routes. Binnen een paar worden swaps en relocaties herhaald tot
    // geen verbetering meer bestaat.
    if (passes <= 0 || neighbor_window <= 0 || search_batches.size() < 2U) {
        return 0;
    }

    std::size_t improvement_count = 0;
    for (int pass = 0; pass < passes && search_batches.size() >= 2U; ++pass) {
        std::vector<std::size_t> sorted_indices(search_batches.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](const std::size_t left, const std::size_t right) {
            return span_center(search_batches[left]) < span_center(search_batches[right]);
        });

        bool improved_this_pass = false;
        for (std::size_t pos = 0; pos < sorted_indices.size(); ++pos) {
            const std::size_t max_neighbor =
                std::min(sorted_indices.size(), pos + 1U + static_cast<std::size_t>(neighbor_window));
            for (std::size_t neighbor_pos = pos + 1U; neighbor_pos < max_neighbor; ++neighbor_pos) {
                SearchBatch& left = search_batches[sorted_indices[pos]];
                SearchBatch& right = search_batches[sorted_indices[neighbor_pos]];
                while (true) {
                    const LocalAction action =
                        find_best_pair_action(instance, items_by_snake_rank, order_items_by_snake_rank, left, right);
                    if (action.type == LocalActionType::none) {
                        break;
                    }
                    apply_local_action(instance, items_by_snake_rank, order_items_by_snake_rank, left, right, action);
                    ++improvement_count;
                    improved_this_pass = true;
                }
            }
        }

        remove_empty_search_batches(search_batches);
        if (!improved_this_pass) {
            break;
        }
    }

    return improvement_count;
}

std::size_t rebuild_nearby_pairs(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    std::vector<SearchBatch>& search_batches,
    int passes,
    int neighbor_window) {
    // Pair rebuild bekijkt dezelfde lokale routeparen, maar verdeelt hun orders
    // volledig opnieuw in plaats van alleen een order te wisselen/verplaatsen.
    if (passes <= 0 || neighbor_window <= 0 || search_batches.size() < 2U) {
        return 0;
    }

    std::size_t rebuild_count = 0;
    for (int pass = 0; pass < passes && search_batches.size() >= 2U; ++pass) {
        std::vector<std::size_t> sorted_indices(search_batches.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](const std::size_t left, const std::size_t right) {
            return span_center(search_batches[left]) < span_center(search_batches[right]);
        });

        bool improved_this_pass = false;
        for (std::size_t pos = 0; pos < sorted_indices.size(); ++pos) {
            const std::size_t max_neighbor =
                std::min(sorted_indices.size(), pos + 1U + static_cast<std::size_t>(neighbor_window));
            for (std::size_t neighbor_pos = pos + 1U; neighbor_pos < max_neighbor; ++neighbor_pos) {
                SearchBatch& left = search_batches[sorted_indices[pos]];
                SearchBatch& right = search_batches[sorted_indices[neighbor_pos]];
                if (rebuild_pair_if_better(instance, items_by_snake_rank, order_items_by_snake_rank, left, right)) {
                    ++rebuild_count;
                    improved_this_pass = true;
                }
            }
        }

        if (!improved_this_pass) {
            break;
        }
    }

    return rebuild_count;
}

std::size_t rebuild_nearby_triples(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    std::vector<SearchBatch>& search_batches,
    int passes,
    int neighbor_window,
    int combination_window) {
    // Triple rebuild is duurder, dus hij blijft lokaal begrensd.
    // Eerst worden vaste gap-patronen getest; daarna extra lokale combinaties
    // binnen combination_window.
    if (passes <= 0 || neighbor_window <= 0 || search_batches.size() < 3U) {
        return 0;
    }

    std::size_t rebuild_count = 0;
    for (int pass = 0; pass < passes && search_batches.size() >= 3U; ++pass) {
        std::vector<std::size_t> sorted_indices(search_batches.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](const std::size_t left, const std::size_t right) {
            return span_center(search_batches[left]) < span_center(search_batches[right]);
        });

        bool improved_this_pass = false;
        for (std::size_t pos = 0; pos + 2U < sorted_indices.size(); ++pos) {
            const std::size_t max_gap =
                std::min<std::size_t>(static_cast<std::size_t>(neighbor_window), (sorted_indices.size() - 1U - pos) / 2U);
            for (std::size_t gap = 1U; gap <= max_gap; ++gap) {
                SearchBatch& first = search_batches[sorted_indices[pos]];
                SearchBatch& second = search_batches[sorted_indices[pos + gap]];
                SearchBatch& third = search_batches[sorted_indices[pos + 2U * gap]];
                if (rebuild_triple_if_better(
                        instance,
                        items_by_snake_rank,
                        order_items_by_snake_rank,
                        first,
                        second,
                        third)) {
                    ++rebuild_count;
                    improved_this_pass = true;
                }
            }

            const std::size_t max_combination_offset = std::min<std::size_t>(
                static_cast<std::size_t>(std::max(0, combination_window)),
                sorted_indices.size() - 1U - pos);
            for (std::size_t first_offset = 1U; first_offset < max_combination_offset; ++first_offset) {
                for (std::size_t second_offset = first_offset + 1U;
                     second_offset <= max_combination_offset;
                     ++second_offset) {
                    if (second_offset == 2U * first_offset) {
                        continue;
                    }

                    SearchBatch& first = search_batches[sorted_indices[pos]];
                    SearchBatch& second = search_batches[sorted_indices[pos + first_offset]];
                    SearchBatch& third = search_batches[sorted_indices[pos + second_offset]];
                    if (rebuild_triple_if_better(
                            instance,
                            items_by_snake_rank,
                            order_items_by_snake_rank,
                            first,
                            second,
                            third)) {
                        ++rebuild_count;
                        improved_this_pass = true;
                    }
                }
            }
        }

        if (!improved_this_pass) {
            break;
        }
    }

    return rebuild_count;
}

std::size_t improve_with_local_search(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<std::vector<int>>& order_items_by_snake_rank,
    std::vector<Batch>& batches,
    int passes,
    int neighbor_window,
    int route_elimination_passes,
    int route_elimination_max_batch_size,
    int route_elimination_beam_width,
    int route_elimination_candidates_per_order,
    int route_elimination_ordering_trials,
    int post_elimination_local_search_passes,
    int pair_rebuild_passes,
    int pair_rebuild_neighbor_window,
    int triple_rebuild_passes,
    int triple_rebuild_neighbor_window,
    int triple_rebuild_combination_window,
    int post_triple_pair_rebuild_passes,
    int post_triple_pair_rebuild_neighbor_window,
    int post_pair_triple_rebuild_passes,
    int post_pair_triple_rebuild_neighbor_window,
    int post_pair_triple_rebuild_combination_window,
    int post_second_triple_pair_rebuild_passes,
    int post_second_triple_pair_rebuild_neighbor_window,
    int final_local_search_passes) {
    // Volledige post-processing pipeline na de savings-greedy:
    // 1. lichte pair-local-search;
    // 2. route-eliminatie;
    // 3. pair rebuild;
    // 4. triple rebuild;
    // 5. extra tail-rondes omdat een grotere rebuild nieuwe pair-kansen opent;
    // 6. finale lichte local search.
    if (passes <= 0 && route_elimination_passes <= 0 && pair_rebuild_passes <= 0 &&
        triple_rebuild_passes <= 0 && post_triple_pair_rebuild_passes <= 0 &&
        post_pair_triple_rebuild_passes <= 0 && post_second_triple_pair_rebuild_passes <= 0) {
        return 0;
    }

    std::vector<SearchBatch> search_batches =
        build_search_batches(instance, items_by_snake_rank, order_items_by_snake_rank, batches);
    std::size_t improvement_count = improve_nearby_pairs(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        search_batches,
        passes,
        neighbor_window);

    const std::size_t eliminated_count = eliminate_routes(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        search_batches,
        route_elimination_passes,
        route_elimination_max_batch_size,
        route_elimination_beam_width,
        route_elimination_candidates_per_order,
        route_elimination_ordering_trials);
    improvement_count += eliminated_count;

    if (eliminated_count > 0U) {
        improvement_count += improve_nearby_pairs(
            instance,
            items_by_snake_rank,
            order_items_by_snake_rank,
            search_batches,
            post_elimination_local_search_passes,
            neighbor_window);
    }

    const std::size_t rebuilt_count = rebuild_nearby_pairs(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        search_batches,
        pair_rebuild_passes,
        pair_rebuild_neighbor_window);
    improvement_count += rebuilt_count;

    const std::size_t triple_rebuilt_count = rebuild_nearby_triples(
        instance,
        items_by_snake_rank,
        order_items_by_snake_rank,
        search_batches,
        triple_rebuild_passes,
        triple_rebuild_neighbor_window,
        triple_rebuild_combination_window);
    improvement_count += triple_rebuilt_count;

    std::size_t post_triple_rebuilt_count = 0;
    if (triple_rebuilt_count > 0U) {
        post_triple_rebuilt_count = rebuild_nearby_pairs(
            instance,
            items_by_snake_rank,
            order_items_by_snake_rank,
            search_batches,
            post_triple_pair_rebuild_passes,
            post_triple_pair_rebuild_neighbor_window);
        improvement_count += post_triple_rebuilt_count;
    }

    std::size_t post_pair_triple_rebuilt_count = 0;
    if (post_triple_rebuilt_count > 0U) {
        post_pair_triple_rebuilt_count = rebuild_nearby_triples(
            instance,
            items_by_snake_rank,
            order_items_by_snake_rank,
            search_batches,
            post_pair_triple_rebuild_passes,
            post_pair_triple_rebuild_neighbor_window,
            post_pair_triple_rebuild_combination_window);
        improvement_count += post_pair_triple_rebuilt_count;
    }

    std::size_t post_second_triple_pair_rebuilt_count = 0;
    if (post_pair_triple_rebuilt_count > 0U) {
        post_second_triple_pair_rebuilt_count = rebuild_nearby_pairs(
            instance,
            items_by_snake_rank,
            order_items_by_snake_rank,
            search_batches,
            post_second_triple_pair_rebuild_passes,
            post_second_triple_pair_rebuild_neighbor_window);
        improvement_count += post_second_triple_pair_rebuilt_count;
    }

    if (rebuilt_count + triple_rebuilt_count + post_triple_rebuilt_count + post_pair_triple_rebuilt_count +
            post_second_triple_pair_rebuilt_count >
        0U) {
        improvement_count += improve_nearby_pairs(
            instance,
            items_by_snake_rank,
            order_items_by_snake_rank,
            search_batches,
            final_local_search_passes,
            neighbor_window);
    }

    batches.clear();
    batches.reserve(search_batches.size());
    for (SearchBatch& batch : search_batches) {
        batches.push_back(std::move(batch.batch));
    }
    return improvement_count;
}

void push_merge_candidate(
    const ProblemInstance& instance,
    const std::vector<int>& items_by_snake_rank,
    const std::vector<ActiveBatch>& batches,
    std::size_t left_id,
    std::size_t right_id,
    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidateLess>& candidates) {
    // Voeg alleen haalbare merges met positieve saving toe.
    // Kandidaten worden lazy gevalideerd wanneer ze uit de queue komen.
    const ActiveBatch& left = batches[left_id];
    const ActiveBatch& right = batches[right_id];
    if (!left.active || !right.active) {
        return;
    }
    if (left.batch.order_indices.size() + right.batch.order_indices.size() > instance.nmax) {
        return;
    }

    const double merged_cost = merged_route_cost(instance, items_by_snake_rank, left.item_mask, right.item_mask);
    const double saving = left.cost + right.cost - merged_cost;
    if (saving <= 1e-9) {
        return;
    }

    candidates.push({saving, merged_cost, left_id, right_id, left.version, right.version});
}

bool is_current_candidate(const std::vector<ActiveBatch>& batches, const MergeCandidate& candidate) {
    // Lazy invalidation: na een merge blijven oude kandidaten in de priority queue,
    // maar hun opgeslagen routeversies komen dan niet meer overeen.
    if (candidate.left_id >= batches.size() || candidate.right_id >= batches.size()) {
        return false;
    }
    const ActiveBatch& left = batches[candidate.left_id];
    const ActiveBatch& right = batches[candidate.right_id];
    return left.active && right.active &&
           left.version == candidate.left_version &&
           right.version == candidate.right_version;
}

}  // namespace

WaveSolveResult solve_wave_greedy(
    const ProblemInstance& instance,
    const BatchEvaluator& evaluator,
    int wave_index,
    const GreedyConfig& config) {
    // Hoofdsolver voor een wave. De wave-grens wordt nooit overschreden:
    // alle batches die hier ontstaan krijgen dezelfde wave_id.
    const auto started = std::chrono::steady_clock::now();
    const Wave& wave = instance.waves.at(static_cast<std::size_t>(wave_index));
    const std::size_t mask_words = (instance.item_ids.size() + 63U) / 64U;
    const std::vector<int> items_by_snake_rank = build_items_by_snake_rank(instance);

    std::vector<ActiveBatch> batches;
    batches.reserve(wave.order_indices.size());
    // Startpunt van savings-greedy: elke order is een afzonderlijke route.
    for (const int order_index : wave.order_indices) {
        Batch batch;
        batch.wave_id = wave.id;
        batch.order_indices.push_back(order_index);
        ActiveBatch active_batch;
        active_batch.item_mask = build_order_mask(instance, order_index, mask_words);
        active_batch.cost = route_cost_for_mask(instance, items_by_snake_rank, active_batch.item_mask);
        active_batch.batch = std::move(batch);
        batches.push_back(std::move(active_batch));
    }

    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidateLess> candidates;
    // Initialiseer alle mogelijke positieve paarmerges.
    for (std::size_t left = 0; left < batches.size(); ++left) {
        for (std::size_t right = left + 1U; right < batches.size(); ++right) {
            push_merge_candidate(instance, items_by_snake_rank, batches, left, right, candidates);
        }
    }

    std::size_t active_count = batches.size();
    std::size_t merge_count = 0;
    // Neem telkens de beste nog actuele saving uit de priority queue.
    while (!candidates.empty()) {
        const MergeCandidate candidate = candidates.top();
        candidates.pop();
        if (!is_current_candidate(batches, candidate)) {
            continue;
        }

        ActiveBatch& left = batches[candidate.left_id];
        ActiveBatch& right = batches[candidate.right_id];
        left.batch.order_indices.insert(
            left.batch.order_indices.end(),
            right.batch.order_indices.begin(),
            right.batch.order_indices.end());
        merge_mask_into(left.item_mask, right.item_mask);
        left.cost = candidate.merged_cost;
        ++left.version;
        right.active = false;
        ++right.version;
        --active_count;
        ++merge_count;

        for (std::size_t other_id = 0; other_id < batches.size(); ++other_id) {
            if (other_id == candidate.left_id || !batches[other_id].active) {
                continue;
            }
            const std::size_t next_left = std::min(candidate.left_id, other_id);
            const std::size_t next_right = std::max(candidate.left_id, other_id);
            push_merge_candidate(instance, items_by_snake_rank, batches, next_left, next_right, candidates);
        }
    }

    std::vector<Batch> result_batches;
    result_batches.reserve(active_count);
    for (const ActiveBatch& batch : batches) {
        if (batch.active) {
            result_batches.push_back(batch.batch);
        }
    }

    std::size_t local_improvements = 0;
    if ((config.local_search_passes > 0 && config.local_search_neighbor_window > 0) ||
        config.route_elimination_passes > 0 ||
        (config.pair_rebuild_passes > 0 && config.pair_rebuild_neighbor_window > 0) ||
        (config.triple_rebuild_passes > 0 && config.triple_rebuild_neighbor_window > 0) ||
        (config.post_triple_pair_rebuild_passes > 0 && config.post_triple_pair_rebuild_neighbor_window > 0) ||
        (config.post_pair_triple_rebuild_passes > 0 && config.post_pair_triple_rebuild_neighbor_window > 0) ||
        (config.post_second_triple_pair_rebuild_passes > 0 &&
         config.post_second_triple_pair_rebuild_neighbor_window > 0)) {
        // De duurdere search-representatie wordt alleen gebouwd als minstens een
        // post-processingfase in de configuratie actief is.
        const std::vector<std::vector<int>> order_items_by_snake_rank = build_order_items_by_snake_rank(instance);
        local_improvements = improve_with_local_search(
            instance,
            items_by_snake_rank,
            order_items_by_snake_rank,
            result_batches,
            config.local_search_passes,
            config.local_search_neighbor_window,
            config.route_elimination_passes,
            config.route_elimination_max_batch_size,
            config.route_elimination_beam_width,
            config.route_elimination_candidates_per_order,
            config.route_elimination_ordering_trials,
            config.post_elimination_local_search_passes,
            config.pair_rebuild_passes,
            config.pair_rebuild_neighbor_window,
            config.triple_rebuild_passes,
            config.triple_rebuild_neighbor_window,
            config.triple_rebuild_combination_window,
            config.post_triple_pair_rebuild_passes,
            config.post_triple_pair_rebuild_neighbor_window,
            config.post_pair_triple_rebuild_passes,
            config.post_pair_triple_rebuild_neighbor_window,
            config.post_pair_triple_rebuild_combination_window,
            config.post_second_triple_pair_rebuild_passes,
            config.post_second_triple_pair_rebuild_neighbor_window,
            config.final_local_search_passes);
    }

    WaveSolveResult result;
    result.batches = std::move(result_batches);
    result.objective = evaluator.wave_cost(wave_index, result.batches);
    result.metadata["construction"] = "savings_merge";
    // Metadata komt niet in de Kaggle-CSV, maar maakt experimenten en paperanalyse
    // reproduceerbaar.
    result.metadata["greedy_choice"] = "largest_positive_pairwise_saving";
    result.metadata["config_id"] = config.config_id;
    result.metadata["merge_updates"] = "lazy_priority_queue";
    result.metadata["merge_count"] = std::to_string(merge_count);
    result.metadata["local_search_passes"] = std::to_string(config.local_search_passes);
    result.metadata["local_search_neighbor_window"] = std::to_string(config.local_search_neighbor_window);
    result.metadata["route_elimination_passes"] = std::to_string(config.route_elimination_passes);
    result.metadata["route_elimination_max_batch_size"] = std::to_string(config.route_elimination_max_batch_size);
    result.metadata["route_elimination_beam_width"] = std::to_string(config.route_elimination_beam_width);
    result.metadata["route_elimination_candidates_per_order"] =
        std::to_string(config.route_elimination_candidates_per_order);
    result.metadata["route_elimination_ordering_trials"] =
        std::to_string(config.route_elimination_ordering_trials);
    result.metadata["post_elimination_local_search_passes"] =
        std::to_string(config.post_elimination_local_search_passes);
    result.metadata["pair_rebuild_passes"] = std::to_string(config.pair_rebuild_passes);
    result.metadata["pair_rebuild_neighbor_window"] = std::to_string(config.pair_rebuild_neighbor_window);
    result.metadata["triple_rebuild_passes"] = std::to_string(config.triple_rebuild_passes);
    result.metadata["triple_rebuild_neighbor_window"] = std::to_string(config.triple_rebuild_neighbor_window);
    result.metadata["triple_rebuild_combination_window"] = std::to_string(config.triple_rebuild_combination_window);
    result.metadata["post_triple_pair_rebuild_passes"] =
        std::to_string(config.post_triple_pair_rebuild_passes);
    result.metadata["post_triple_pair_rebuild_neighbor_window"] =
        std::to_string(config.post_triple_pair_rebuild_neighbor_window);
    result.metadata["post_pair_triple_rebuild_passes"] =
        std::to_string(config.post_pair_triple_rebuild_passes);
    result.metadata["post_pair_triple_rebuild_neighbor_window"] =
        std::to_string(config.post_pair_triple_rebuild_neighbor_window);
    result.metadata["post_pair_triple_rebuild_combination_window"] =
        std::to_string(config.post_pair_triple_rebuild_combination_window);
    result.metadata["post_second_triple_pair_rebuild_passes"] =
        std::to_string(config.post_second_triple_pair_rebuild_passes);
    result.metadata["post_second_triple_pair_rebuild_neighbor_window"] =
        std::to_string(config.post_second_triple_pair_rebuild_neighbor_window);
    result.metadata["final_local_search_passes"] = std::to_string(config.final_local_search_passes);
    result.metadata["local_improvements"] = std::to_string(local_improvements);

    const auto finished = std::chrono::steady_clock::now();
    result.runtime_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    return result;
}

}  // namespace optioryx
