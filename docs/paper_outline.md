# Paper Outline

## 1. Introductie
- Probleemstelling: Order Batching Problem met vaste snake-pickvolgorde.
- Context van Optioryx en magazijnoptimalisatie.
- Doel: totale pickafstand minimaliseren binnen wave- en capaciteitsconstraints.
- Belang van optimalisatie: lagere wandeltijd, snellere orderverwerking.

## 2. Analyse Van De Oorspronkelijke Oplossing
- Greedy savings als eerste werkende basis.
- BatchEvaluator voor routekosten en validatie.
- Input via fixtures of competitiedata.
- Output als Kaggle-submission CSV.
- Beperkingen: veel pair-evaluaties, nog beperkte local search, routecount niet expliciet geoptimaliseerd.

## 3. Bedrijfsfeedback
- Submissionvalidatie lokaal uitvoeren.
- `LIST_ID` mag geen waves mengen.
- Maximaal 20 orders per lijst.
- Greedy warm-start is belangrijk.
- Denk in overlap/Jaccard en snake-spreiding.
- Test rijkere neighbourhoods dan enkel swap.
- Vergelijk runtime versus kwaliteit.

## 4. Onderzochte Alternatieven
- Savings greedy baseline.
- Simulated annealing met greedy/random start, cooling schedules en meerdere neighbourhoods.
- Genetisch algoritme met permutatierepresentatie en DP-decoder.
- Seed-and-selection heuristiek geinspireerd door literatuur.
- Post-processing local search en route-eliminatie.

## 5. Implementatie Van Verbeteringen
- Lazy priority queue voor greedy savings.
- Item masks voor snelle routekostberekening in hot loops.
- Bounded local search met swaps en relocaties tussen snake-nabije batches.
- Merge-split neighbourhood voor simulated annealing.
- Route-eliminatie als post-processing.
- Best-improvement route-eliminatie met beperkte beam search.
- Multi-ordering beam assignment als getest maar niet gekozen alternatief.
- Twee-route rebuild om routecompositie te verbeteren nadat het minimumaantal routes bereikt is.
- Regret-greedy assignment binnen twee-route rebuild.
- Drie-route rebuild als grotere deterministische neighbourhood.
- Post-triple pair rebuild als lokale verfijning na grotere moves.
- Tweede triple-rebuild-fase als laatste grotere polish na pair-verfijning.
- Combinatorische triple-vensters als uitbreiding op de eerdere sparse arithmetic-gap triple scan.
- Tail pair-rebuild na de tweede triple-fase als laatste twee-route verfijning.
- Finale local-search polish na rebuilds.
- Validatietests voor veelvoorkomende Kaggle-fouten.

## 6. Evaluatie En Vergelijking
- Testopstelling: Release build, lokale competitiedata.
- Correctheid: `ctest` en `validate-submission`.
- Kwaliteit: lokale score.
- Snelheid: wall-clock runtime via PowerShell `Measure-Command`.
- Schaalbaarheid: runtime op full dataset en rooktests op fixtures.
- Tabellen: zie `docs/algorithm_comparison.md`.

## 7. Gekozen Eindoplossing
- `greedy_competition_final`: beam route-eliminatie, acht regret-aware pair-rebuild-passen, drie triple-rebuild-passen, twee post-triple pair-rebuild-passen, twee tweede triple-passen, combinatorische triple-vensters van 11, drie tail pair-rebuild-passen en finale local-search polish.
- Reden: beste lokaal gemeten score, valide output, deterministisch en goed uitlegbaar.
- Trade-off: veel hogere runtime dan fast greedy en beam-only, maar duidelijk betere score.

## 8. Discussie
- Sterktes: deterministic, feedbackgericht, valide en meetbaar beter.
- Zwaktes: route-eliminatie en rebuilds blijven heuristisch en runtime is hoog.
- Beperkingen: SA/GA gebruiken nog geen snelle delta-evaluatie.
- Mogelijke verbetering: grotere triple-vensters, min-cost-flow herverdeling of ruin-and-recreate met sterke acceptatiecriteria.

## 9. Conclusie
- Samenvatting van scoreverbetering.
- Verwerkte feedback.
- Waarom de gekozen aanpak geschikt is voor de competitie en verdedigbaar in het verslag.
