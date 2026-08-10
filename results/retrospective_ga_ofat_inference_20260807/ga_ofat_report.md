# Gekoppelde inferentie voor de GA-OFAT-studie

De archiefdata bevat negen configuraties, tien gedeelde seeds per configuratie en twaalf waves per volledige run. Elk contrast is berekend als variant min basis. Voor de doelfunctie betekent een negatieve waarde dus een verbetering. De statistische eenheid is een volledige instance na sommering over alle waves.

## Kwaliteit en uitvoeringstijd

| factor | variant (basis) | $\Delta D$ gemiddeld [95%-BI] | ruwe p | Holm p | $\Delta t$ gemiddeld [95%-BI] | Holm p tijd |
|---|---:|---:|---:|---:|---:|---:|
| elitisme | 0 (0.08) | 49.2 [-80.8, 179.2] | 0.414342 | 1 | 17.31% [4.89, 29.73]% | 0.03401 |
| elitisme | 0.2 (0.08) | -12.7 [-188.4, 163.0] | 0.873542 | 1 | -5.76% [-7.66, -3.87]% | 0.000422533 |
| generaties | 10 (20) | 80.8 [-13.4, 175.0] | 0.0841491 | 0.589044 | -15.21% [-17.23, -13.18]% | 5.79669e-07 |
| generaties | 40 (20) | -53.9 [-230.0, 122.3] | 0.506561 | 1 | 21.29% [18.68, 23.89]% | 2.65888e-08 |
| mutatiekans | 0.05 (0.2) | 47.7 [-84.8, 180.1] | 0.436379 | 1 | -2.86% [-5.59, -0.12]% | 0.0844796 |
| mutatiekans | 0.4 (0.2) | -39.5 [-193.7, 114.7] | 0.576767 | 1 | 0.61% [-2.71, 3.94]% | 0.713211 |
| populatie | 24 (12) | -243.3 [-477.7, -9.0] | 0.0433881 | 0.347105 | 55.25% [51.93, 58.57]% | 1.02847e-11 |
| populatie | 6 (12) | 48.2 [-83.5, 179.8] | 0.42906 | 1 | -26.32% [-27.96, -24.68]% | 4.272e-09 |

Na Holm-correctie over de acht kwaliteitscontrasten verschilt geen variant significant van de basis bij 5%.

## Diversiteit

Bij de basisconfiguratie daalt de gemiddelde genormaliseerde positionele Hammingdiversiteit van 0.997 bij generatie 0 naar 0.109 bij generatie 20. Daarmee blijft gemiddeld 11.0% van de initiële diversiteit over.

De einddiversiteit is per seed eerst over de twaalf waves gemiddeld. Voor elk variant-basiscontrast staat de volledige gekoppelde inferentie in `ga_ofat_contrasts.csv`. De tabel `ga_ofat_diversity_summary.csv` bevat daarnaast de beginwaarde, eindwaarde, verandering en behouden fractie voor alle negen configuraties.

## Afbakening

Deze OFAT-resultaten zijn conditioneel op order crossover, tournamentselectie met grootte 3 en een 50/50-mix van insertion- en swapmutatie. Voor populatie-, generatie- en mutatiekanscontrasten bleef het elitisme 0,08. Alleen de twee elitismecontrasten wijzigden die fractie naar 0 of 0,20. Interacties tussen factoren zijn niet identificeerbaar uit een OFAT-opzet. De runtimevolgorde werd in de oorspronkelijke suite niet gerandomiseerd, waardoor de tijdscontrasten ook systeem- en volgorde-effecten kunnen bevatten.

De ruwe p-waarden zijn tweezijdige gekoppelde Student-t-toetsen. Holm-correctie is afzonderlijk toegepast op drie families van telkens acht vooraf bepaalde contrasten: doelfunctie, interne solverruntime en einddiversiteit.
