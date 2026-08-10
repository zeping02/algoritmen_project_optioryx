# Gekoppelde inferentie voor het SA-factorieel experiment

De analyse omvat acht configuraties uit een volledig 2x2x2-ontwerp en tien gedeelde seeds. Per configuratie en seed zijn de twaalf waves eerst opgeteld. De zeven contrasten zijn daarna per seed gevormd.

Een hoofdeffect is het gemiddelde verschil tussen het alternatieve en het referentieniveau. Interacties zijn gekoppelde verschillen-van-verschillen met dezelfde plus/min-codering.

| effect | richting | gemiddeld effect op D [95%-BI] | ruwe p | Holm p | gemiddeld effect op tijd in ms [95%-BI] | Holm p tijd |
|---|---|---:|---:|---:|---:|---:|
| cooling | lundy_mees-geometric | -744.4 [-9 200.4, 7 711.5] | 0.846567 | 1 | 213.8 [-77.7, 505.3] | 0.334181 |
| neighborhood | swap-relocate | -204 701.9 [-211 955.7, -197 448.0] | 2.86596e-13 | 1.6772e-12 | -263.7 [-571.0, 43.6] | 0.334181 |
| start_strategy | randomized-greedy | 5 450 753.6 [5 442 863.4, 5 458 643.8] | 9.15928e-26 | 6.4115e-25 | -12 385.6 [-12 653.4, -12 117.7] | 2.37047e-14 |
| cooling:neighborhood | lundy_mees-geometric:swap-relocate | -15 511.1 [-29 476.7, -1 545.4] | 0.033175 | 0.1327 | -566.4 [-1 145.9, 13.0] | 0.320132 |
| cooling:start_strategy | lundy_mees-geometric:randomized-greedy | -1 235.1 [-18 181.4, 15 711.3] | 0.872691 | 1 | -427.7 [-1 011.0, 155.6] | 0.334181 |
| neighborhood:start_strategy | swap-relocate:randomized-greedy | -408 236.2 [-422 662.4, -393 810.0] | 2.79533e-13 | 1.6772e-12 | 527.2 [-85.8, 1 140.2] | 0.334181 |
| cooling:neighborhood:start_strategy | lundy_mees-geometric:swap-relocate:randomized-greedy | -31 111.5 [-59 457.5, -2 765.6] | 0.0348273 | 0.1327 | 1 136.7 [-20.3, 2 293.7] | 0.320132 |

Na Holm-correctie blijven de volgende kwaliteitscontrasten significant op het 5%-niveau: neighborhood, start_strategy, neighborhood:start_strategy.

De Holm-correctie is afzonderlijk toegepast op de zeven kwaliteitscontrasten en de zeven runtimecontrasten. De toets is telkens een tweezijdige gekoppelde Student-t-toets op de tien seedverschillen.

Positieve hoofdeffecten op de doelfunctie betekenen dat het alternatieve niveau gemiddeld slechter scoort, aangezien een lagere afstand beter is. Interactiecontrasten vereisen een conditionele interpretatie en zijn geen afzonderlijke algoritmevariant.
