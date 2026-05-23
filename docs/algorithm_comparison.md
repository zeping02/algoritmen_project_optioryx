# Algorithm Comparison

Local measurements were run on the included competition data in a Release build. Runtime depends on the machine, so use the scores and relative trends as the most stable comparison.

| Version | Approach | Advantages | Disadvantages | Correctness | Speed | Scalability | Why Chosen Or Not |
|---|---|---|---|---|---|---|---|
| Example submission | Provided baseline grouping | Valid reference, easy sanity check | Very weak score | Validated | Very fast to score | Not an optimizer | Only useful as baseline |
| Greedy fast | Savings merge with lazy priority queue, no local search | Simple, deterministic, explainable | Leaves many local improvements | Validated | About 20s | Good enough for full data | Good baseline, not final |
| Greedy local | Savings + bounded pairwise swaps/relocations | Large score gain with small added complexity | Still limited to nearby batch pairs | Validated | About 22s | Good | Useful balanced config |
| Greedy strong/ultra | Wider local search | Better quality | Longer runtime | Validated | About 26-43s | Moderate | Good quality/runtime trade-off |
| Greedy best_plus | Wider pairwise descent | Best score before route elimination | Still 534 routes | Validated | About 81s | Moderate | Previous best |
| Greedy eliminate | best_plus + route elimination | Fewer routes and lower score | More runtime, greedy insertion can be myopic | Validated | About 95s | Moderate | Good stepping stone |
| Greedy eliminate polished plus | Route elimination plus second local-search polish | Best measured score, fewer routes, still deterministic | Diminishing returns and higher runtime | Validated | About 121s | Moderate | Current best for competition submission |
| Greedy final | Wider window and more passes | Tests whether local search is saturated | No extra score improvement | Validated | About 104s | Moderate | Not chosen because score matched polished plus |
| Greedy beam eliminate | Best-improvement route elimination with beam assignment | First run to reach theoretical minimum of 517 routes | Higher runtime and still locally optimal route composition | Validated | About 135s | Moderate | Important stepping stone, not final |
| Greedy beam eliminate strong | Larger beam and more candidates | Tests whether a broader beam helps | Worse score than smaller beam | Validated | About 116s | Moderate | Not chosen |
| Greedy beam multi-order | Beam elimination with multiple source-ordering trials | Tests whether insertion order in route elimination matters | Slightly worse score than single ordering | Validated | About 93s | Moderate | Not chosen, useful negative result |
| Pair rebuild fast | Beam eliminate plus one two-route rebuild pass with window 8 | Breaks out of single-order swap/relocate local optimum | Slower than beam-only and not the best score | Validated | About 194s | Moderate | Good speed/quality trade-off, not final |
| Pair rebuild xwide | One two-route rebuild pass with window 40 | Large score gain while keeping 517 routes | More runtime than beam-only | Validated | About 310s | Moderate | Shows route composition, not route count, became bottleneck |
| Pair rebuild five-pass polished | Five rebuild passes plus final local search | Strong quality gain and still deterministic | Long runtime | Validated | About 724s | Lower for production, acceptable for offline competition | Strong balanced high-quality variant |
| Pair rebuild six-pass polished | Six rebuild passes plus final local search | Best score before regret-greedy assignment | Very long runtime and small gain over five-pass polished | Validated | About 853s | Lower for production, acceptable for offline competition | Superseded by regret-greedy rebuild |
| Regret-greedy pair rebuild | Pair rebuild also assigns most constrained orders first by insertion-cost regret | Strong score gain, deterministic, faster in measured final run | Still a long offline heuristic and ninth pass was worse | Validated | About 487s | Lower for production, acceptable for offline competition | Important stepping stone |
| Three-route rebuild | Rebuilds three snake-nearby routes and keeps only strict total-cost improvements | Best measured score and crosses below 5 million | Extra implementation complexity and longer runtime than pair-only | Validated | About 565s | Lower for production, acceptable for offline competition | Current final score-oriented config |
| Post-triple pair rebuild | Reruns pair rebuild after three-route moves | Best measured score; catches pair improvements opened by triple moves | Extra runtime and third pass was worse | Validated | About 671s | Lower for production, acceptable for offline competition | Current final score-oriented config |
| Second triple phase | Reruns triple rebuild after post-triple pair polish | Best measured score; shows pair and triple moves are complementary | Additional runtime; third pass no longer improved | Validated | About 704s | Lower for production, acceptable for offline competition | Current final score-oriented config |
| Simulated annealing | Greedy start plus relocate/swap/merge-split moves | Required by assignment, useful comparison | Short runs did not beat improved greedy | Validated in tests | Depends on iterations | Needs delta evaluation for long runs | Kept for analysis, not current final submission |
| Genetic algorithm | Permutation representation with DP decoder | Required by assignment, preserves feasible batches | Slower and did not beat improved greedy in fast configs | Validated in tests | Slowest tested | Needs stronger tuning/delta reuse | Kept for analysis, not current final submission |

## Best Measured Competition Result

```text
build_release/optioryx_cli solve --solver greedy --input data --output build_release/greedy_final.csv --config configs/greedy_competition_final.cfg
build_release/optioryx_cli validate-submission --input data --submission build_release/greedy_final.csv
```

Result:

```text
Valid submission: yes
Local score: 4.99368e+06
Routes: 517
```

## Paper Use

This table can support the evaluation section: it shows that the final choice was not arbitrary. The final solver is selected because it gives the best measured local score while remaining deterministic and explainable. The five-pass polished rebuild config is a useful alternative when runtime matters more than the last few hundred score points.
