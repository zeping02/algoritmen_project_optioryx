# Per-wave-analyse van de Release-budgetstudie

Deze analyse gebruikt uitsluitend bestaande solverresultaten. Er zijn geen nieuwe
optimalisatieruns uitgevoerd.

## Validatie en protocol

- Gevalideerde records: 360 = 12 waves × 3 methoden × 10 seeds.
- Alle geselecteerde runs zijn succesvol, niet getime-out en haalbaar.
- Iedere SA-10000- en GA-40-run is gekoppeld aan greedy met dezelfde wave en seed.
- Absolute winst is `D_greedy - D_kandidaat`; een positieve waarde is beter.
- Procentuele winst wordt per gekoppelde seed berekend als
  `100 * (D_greedy - D_kandidaat) / D_greedy`.
- Gemiddelde, steekproef-SD en tweezijdig Student-t-95%-BI gebruiken de tien
  gekoppelde seeds binnen één wave.
- De tweezijdige t-toets gebruikt H0: gemiddelde gekoppelde winst = 0.
- Holm corrigeert per kandidaat over de twaalf wave-toetsen bij alpha=0.05.
- Waves zijn alleen de twaalf eenheden van de exploratieve Spearman-analyse en
  worden niet als extra seedreplicaten behandeld.

## Kwaliteitsbevindingen

### SA-10000

- Minstens één seed verbetert in 11/12 waves.
- Een ruw tweezijdig 95%-BI sluit nul positief uit in 8/12 waves.
- Na Holm-correctie blijven 4/12 waves significant.
- Grootste gemiddelde relatieve winst: W000011 met 1.184080%.
- Grootste gemiddelde absolute winst: W000005 met 1259.826 afstandseenheden.

### GA-40

- Minstens één seed verbetert in 5/12 waves.
- Een ruw tweezijdig 95%-BI sluit nul positief uit in 0/12 waves.
- Na Holm-correctie blijven 0/12 waves significant.
- Grootste gemiddelde relatieve winst: W000011 met 0.226428%.
- Grootste gemiddelde absolute winst: W000012 met 106.361 afstandseenheden.

### Holm-significante waves

| methode | wave | gem. winst | 95%-BI | gem. winst (%) | Holm-p |
|---|---:|---:|---:|---:|---:|
| SA-10000 | W000002 | 269.590 | [141.819, 397.361] | 0.038480 | 0.01011063 |
| SA-10000 | W000005 | 1259.826 | [646.606, 1873.046] | 0.281420 | 0.01085777 |
| SA-10000 | W000007 | 445.033 | [312.622, 577.445] | 0.064677 | 0.00036477 |
| SA-10000 | W000010 | 1220.471 | [1016.696, 1424.247] | 0.178304 | 0.00000327 |

## Exploratieve wavekenmerken

Spearman gebruikt de gemiddelde gekoppelde procentuele winst per wave. De drie
vooraf bestaande kenmerken zijn orderaantal, gemiddelde paarsgewijze
Jaccard-overlap en SD van de orderspans. Holm corrigeert per kandidaat over deze
drie associaties. Leave-one-wave-out is een gevoeligheidsanalyse, geen BI.

| methode | kenmerk | rho | ruwe p | Holm-p | LOO-bereik |
|---|---|---:|---:|---:|---:|
| SA-10000 | order_count | -0.283713 | 0.37150967 | 0.74301933 | [-0.514808, -0.068337] |
| SA-10000 | overlap_density | 0.279720 | 0.37856867 | 0.74301933 | [0.063636, 0.563636] |
| SA-10000 | span_dispersion | -0.384615 | 0.21702023 | 0.65106068 | [-0.645455, -0.236364] |
| GA-40 | order_count | -0.773355 | 0.00317307 | 0.00951922 | [-0.909166, -0.696514] |
| GA-40 | overlap_density | 0.421092 | 0.17280996 | 0.34561992 | [0.221114, 0.610696] |
| GA-40 | span_dispersion | -0.280728 | 0.37678084 | 0.37678084 | [-0.456027, -0.183402] |

### Voorzichtige interpretatie

- GA-40: `order_count` heeft rho=-0.773 (Holm-p=0.009519); de LOO-rho blijft tussen -0.909 en -0.697.

Met slechts twaalf waves, gebonden winstwaarden en meerdere nulwaarden zijn deze
associaties beschrijvend. Ze bewijzen geen causaal wave-effect.
