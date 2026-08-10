# Long-convergence summary

All statistics use the complete-instance best-so-far curve obtained by summing the twelve wave curves at each step.

| config | seeds | no gain | t95 median [IQR], improvers | t99 median [IQR], improvers | last improvement median | improved in last 10% | edge improvements | solver time mean (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sa_i10000 | 10 | 0 | 9259.0 [8597.5, 9338.5] | 9414.5 [9255.5, 9593.8] | 9853.5 | 10 | 0 | 13.79 |
| sa_i50000 | 10 | 0 | 42871.5 [41143.2, 45999.8] | 47924.0 [46263.0, 49190.5] | 49388.5 | 10 | 0 | 16.30 |
| ga_g40 | 10 | 5 | 28.0 [18.0, 39.0] | 28.0 [18.0, 39.0] | 28.0 | 2 | 1 | 32.32 |
| ga_g100 | 10 | 1 | 83.0 [75.0, 90.0] | 84.0 [82.0, 92.0] | 84.0 | 4 | 1 | 53.37 |
| ga_g200 | 10 | 1 | 164.0 [145.0, 172.0] | 164.0 [145.0, 172.0] | 164.0 | 1 | 0 | 88.88 |

Definitions: t95/t99 are the first steps reaching 95%/99% of each run's eventual within-budget gain. Runs without a strict gain have blank t95/t99. An edge improvement occurs at the final allowed step.

Exact seconds-to-t95/t99 are unavailable because convergence points carry no per-step timestamps. Only the exact step and complete-run/per-wave times are recorded.

SA 10k and SA 50k independently cool from 60 to 0.05 with budget-specific geometric factors. They are paired alternative schedules, not a 10k prefix continued for another 40k iterations.

GA uses one RNG stream across the twelve sequential waves. Changing the generation horizon changes how many draws wave 1 consumes, so later waves start from different RNG states. The 40/100/200 full-instance curves are therefore same-seed alternative horizons, not continuations of one shared full-instance trajectory. Objective-prefix equality is audited descriptively.

## Paired budget contrasts

| short | long | kind | n | mean added gain [95% CI] | mean runtime ratio [95% CI] |
|---|---|---|---:|---:|---:|
| sa_i10000 | sa_i50000 | independent_budget_specific_schedules | 10 | 13605.920 [12070.824, 15141.017] | 1.184 [1.144, 1.223] |
| ga_g40 | ga_g100 | same_seed_separate_generation_horizons | 10 | 438.334 [69.241, 807.427] | 1.653 [1.607, 1.699] |
| ga_g100 | ga_g200 | same_seed_separate_generation_horizons | 10 | 537.975 [-113.275, 1189.226] | 1.666 [1.618, 1.714] |
| ga_g40 | ga_g200 | same_seed_separate_generation_horizons | 10 | 976.309 [456.150, 1496.468] | 2.753 [2.657, 2.849] |
