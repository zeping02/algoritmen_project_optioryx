#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace optioryx {

// Centrale datastructuren van het project.
//
// Belangrijk idee:
// - een order hoort altijd bij exact een wave;
// - een batch/picklijst mag alleen orders uit een wave bevatten;
// - de solver werkt intern met integer-indices voor snelheid, maar output gebruikt de originele ids.

// Een order bestaat uit zijn originele ORDER_ID, de wave waartoe hij behoort,
// en de lijst van picklocaties als indices in ProblemInstance::item_ids.
struct Order {
    std::string id;
    std::string wave_id;
    std::vector<int> items;
};

// Een wave groepeert alle orders die samen bekeken mogen worden.
// Orders uit verschillende waves mogen nooit in dezelfde batch terechtkomen.
struct Wave {
    std::string id;
    std::vector<int> order_indices;
};

// Een batch is een picklijst binnen een wave.
// order_indices verwijst naar ProblemInstance::orders.
struct Batch {
    std::string wave_id;
    std::vector<int> order_indices;
};

// Meetpunt voor iteratieve algoritmes zoals simulated annealing en genetic search.
// control_value is bewust generiek: bij SA is dit de temperatuur, bij GA de diversiteit.
struct ConvergencePoint {
    int step = 0;
    double best_objective = 0.0;
    double current_objective = 0.0;
    double control_value = 0.0;
};

// Resultaat voor een enkele wave, inclusief metadata voor analyse en paper.
struct WaveSolution {
    std::string wave_id;
    std::vector<Batch> batches;
    double objective = 0.0;
    double runtime_ms = 0.0;
    std::vector<ConvergencePoint> convergence;
    std::map<std::string, std::string> metadata;
};

// Volledige oplossing over alle waves heen.
// feasible en validation_errors worden na het oplossen gevuld door de evaluator.
struct Solution {
    std::string solver_name;
    std::string config_id;
    std::uint64_t seed = 0;
    std::vector<WaveSolution> wave_solutions;
    double objective = 0.0;
    double runtime_ms = 0.0;
    bool feasible = false;
    std::vector<std::string> validation_errors;
};

// In-memory representatie van de opdrachtdata.
//
// snake_rank geeft de vaste volgorde waarin locaties worden gepickt.
// distances bevat de afstand tussen elke twee locaties.
// De *_index_by_id maps maken het mogelijk om CSV-ids eenmalig naar snelle indices om te zetten.
struct ProblemInstance {
    std::string name;
    std::size_t nmax = 0;
    int start_index = 0;
    int end_index = 0;
    std::vector<std::string> item_ids;
    std::vector<int> snake_rank;
    std::vector<std::vector<double>> distances;
    std::vector<Order> orders;
    std::vector<Wave> waves;
    std::unordered_map<std::string, int> item_index_by_id;
    std::unordered_map<std::string, int> order_index_by_id;
    std::unordered_map<std::string, int> wave_index_by_id;
};

// Extra analysekenmerken per wave, gebruikt om benchmarkresultaten te interpreteren.
struct WaveMetrics {
    double overlap_density = 0.0;
    double span_dispersion = 0.0;
};

}  // namespace optioryx
