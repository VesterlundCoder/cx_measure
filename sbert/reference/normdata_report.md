# Test av C\_X – konceptuell koherens på verkliga normdata

Nov 12, 2026 · @David Vesterlund

## Sammanfattning

Måttet C\_X, beräknat exakt som Osta-Vélez och Gärdenfors (2026) definierar det, mäter i praktiken hur långt en kategori ligger från "saker i allmänhet", inte hur starkt dess egenskaper samvarierar. På 15 kategorier ur Leuvendatabasen och 10 ur McRaes normer får det följande stöd:

1. **C\_X är ett särpräglingsmått.** Över 11 kategorier korrelerar C\_X med den andel av matrisen som bara beror på avståndet till den universella prototypen med Spearman ρ = 0,93 (McRae: ρ = 0,94). En syntetisk kategori helt utan samvariation men långt från prototypen får C\_X = 0,48, lika högt som en kategori med mycket stark samvariation (0,46).
2. **Naturliga arter är inte mer koherenta än artefakter.** Skillnaden är liten och icke-signifikant i alla tolv varianter (p ≥ 0,21), trots att människors likhetsbedömningar visar en tydlig skillnad i den riktning artikeln förutsäger (0,45 mot 0,30, p = 0,004).
3. **Överordnade begrepp får lägre C\_X**, som artikeln förutsäger. Men i den rena samvariationsversionen är 6 av 11 verkliga kategorier inte mer koherenta än slumpvis hopblandade grupper av samma storlek. Fågel, däggdjur, verktyg och kläder ligger under slumpmedianen.
4. **Människors bedömda likhet inom kategorin följer inte artikelns C\_X** (ρ = −0,12). Den version författarna förkastar, med kategorins eget medelvärde, ger en måttlig men osäker korrelation (ρ = 0,42–0,52, p = 0,05–0,20).
5. **Typikalitet förutsägs väl av artikelns första principalkomponent** (medel-ρ = 0,72, positivt i 11 av 11 kategorier). Men komponenten pekar nästan exakt mot kategorins mittpunkt (cosinus 0,996) och ger samma rangordning som en enkel kontrastmodell (ρ = 0,99). Den rena samvariationskomponenten förutsäger ingenting (ρ = 0,06).

Slutsatsen är att det som fungerar i måttet är dess kontrastterm, som liknar Rosch och Mervis familjelikhet, och inte den samvariationsstruktur som artikeln bygger sin teori på. Inga data med direkta koherensbetyg fanns tillgängliga, så punkt 4 bygger på ett närliggande mått.

## Fråga och data

Artikeln påstår att ett begrepp är koherent i den mån dess egenskaper samvarierar så att de kan komprimeras till få dimensioner. Ur det härleder den att naturliga arter är mer koherenta än artefakter (H1), att överordnade begrepp är mindre koherenta än underordnade (H2), att koherens ska stämma med hur sammanhållen en kategori uppfattas (H3), och att exemplar som följer kategorins huvudmönster upplevs som typiska (H4). Måttet beräknas aldrig på data i artikeln.

| Dataset | Innehåll som används | Kategorier |
| --- | --- | --- |
| [Leuven Natural Concept Database](https://github.com/mikulatomas/dutch-concepts) (De Deyne m.fl. 2008) | Exemplar × egenskap-matriser där 4 bedömare angett om egenskapen gäller (0–4); viktighetsbetyg per egenskap (−3 till 3); typikalitetsbetyg; parvisa likhetsbedömningar | 15: fågel, fisk, insekt, däggdjur, reptil; kläder, köksredskap, musikinstrument, verktyg, fordon, vapen; frukt, grönsak, yrke, sport |
| [McRae m.fl. 2005](https://github.com/div5yesh/semantic-feature-extraction) | Produktionsfrekvens för 2 526 egenskaper hos 541 begrepp; kategoritillhörighet från taxonomiska egenskaper ("a\_bird") | 10: däggdjur, fågel, fisk, frukt, grönsak, vapen, musikinstrument, verktyg, köksredskap, fordon |

Leuvendatan är huvudtestet eftersom den har fullständiga bedömningar för varje egenskap och varje exemplar, plus människobetyg att jämföra med. Viktighetsbetygen ger salience-vikter som inte är godtyckliga, vilket löser ett av artikelns öppna problem. McRaes normer bygger på fri produktion och är glesa, så de fungerar som replikation för H1 och H2.

Direkta betyg av typen "hur väl hänger den här kategorin ihop?" finns inte i något av dataseten. H3 använder därför medelvärdet av människors bedömda likhet mellan kategorins exemplar.

## Metod

C\_X implementerades exakt enligt artikelns avsnitt 3.3. Varje dimension multipliceras med sin salience-vikt s, och matrisen beräknas runt den universella prototypen u i stället för runt kategorins eget medelvärde:

```latex
S_u = \frac{1}{n}\sum_k \big(s \odot (x_k - u)\big)\big(s \odot (x_k - u)\big)^\top, \qquad p_i = \frac{\lambda_i}{\sum_j \lambda_j}, \qquad C_X = \sum_i w_i\, p_i, \quad w_i \propto (1-r)\,r^{\,i-1}
```

Matrisen kring u kan skrivas om som summan av en ren samvariationsdel och en avståndsdel, där m är avståndet från prototypen till kategorins mittpunkt:

```latex
S_u = \mathrm{Cov}(X) + m\,m^\top, \qquad m = s \odot (\bar{x} - u)
```

Andelen tr(m mᵀ) / tr(S\_u) kallas här *avståndsandel*. Den mäter hur mycket av C\_X som bara beror på att kategorin ligger långt från allt annat.

**Operationaliseringar.**

- Salience = medelvärdet av viktighetsbetygen, linjärt omskalat från −3..3 till 0..1.
- Universell prototyp = medelvärdet av alla exemplar i domänen: 129 djur respektive 166 artefakter. Domänerna har olika egenskapslistor, så en gemensam prototyp för "sak" går inte att definiera.
- Huvudspecifikationen använder r = 0,5. Robustheten testades för r = 0,3; 0,5; 0,7 och 0,9.
- Domänmatrisen används för de 11 domänkategorierna, eftersom samma bedömargrupp då står för alla exemplar. Kategorimatriserna kommer från andra bedömare och stämmer bara ungefär med domänmatrisen (r ≈ 0,8). De används för jämförelsen över alla 15 kategorier.

**Varianter.** Artikelns version (universell prototyp, salience-viktad) jämförs med den rena samvariationsversionen (kategorins eget medelvärde), med uniforma vikter och med hela domänens egenskapsrum.

**Kontroller.**

- Storleksmatchning: kategorierna har 20–33 exemplar, och C\_X beror på antalet. Alla kategorier delsamplades därför till n = 20.
- Slumpkategorier: för varje verklig kategori drogs 500 slumpvisa grupper av samma storlek ur hela domänen.
- Exakta permutationstest användes för skillnaden mellan natur och artefakt (462 möjliga uppdelningar).
- Bootstrap med 1 000 dragningar gav konfidensintervall per kategori.
- Jämförelsemodeller för typikalitet: Rosch och Mervis familjelikhet, cosinus mot prototypen, och en kontrastmodell som projicerar varje exemplar på riktningen m.

## Resultat

### Origo-artefakten: C\_X mäter avstånd

Artikelns C\_X följer avståndsandelen nästan perfekt: ρ = 0,93 över 11 Leuvenkategorier (p < 0,001) och ρ = 0,94 över 10 McRaekategorier. Sambandet håller för alla r (ρ = 0,84–0,94). För den rena samvariationsversionen är korrelationen med avståndsandelen bara ρ = 0,21. Fisk och musikinstrument får högst C\_X för att de ligger längst från sin domäns prototyp, inte för att deras egenskaper samvarierar mer.

Det syntetiska testet visar samma sak isolerat (30 exemplar, 20 dimensioner):

| Syntetisk kategori | C\_X, artikelns version | C\_X, ren samvariation | Avståndsandel |
| --- | --- | --- | --- |
| Ingen samvariation, nära prototypen | 0,126 | 0,125 | 0,04 |
| Ingen samvariation, långt från prototypen | 0,478 | 0,131 | 0,95 |
| Stark samvariation, nära prototypen | 0,456 | 0,456 | 0,05 |
| Stark samvariation, långt från prototypen | 0,497 | 0,456 | 0,94 |

### H1: naturliga arter mot artefakter

Ingen variant av C\_X skiljer djur från artefakter, men människorna gör det. Medelvärden över 5 djur- och 6 artefaktkategorier, exakt permutationstest:

| Mått | Djur | Artefakter | p |
| --- | --- | --- | --- |
| C\_X, artikelns version | 0,320 | 0,342 | 0,43 |
| C\_X, ren samvariation | 0,203 | 0,188 | 0,51 |
| C\_X, storleksmatchad (n = 20) | 0,324 | 0,347 | 0,39 |
| Familjelikhet (cosinus) | 0,799 | 0,750 | 0,12 |
| Människors bedömda likhet | 0,449 | 0,302 | 0,004 |

Resultatet håller för alla r och vikter: inga av de tolv varianterna har p under 0,21. I McRaedata går den rena samvariationsversionen åt fel håll (djur 0,124, artefakter 0,167, p = 0,14).

### H2: överordnade mot underordnade begrepp

C\_X för "djur" som helhet är 0,145, lägre än för alla 5 djurkategorier. För "artefakt" är det 0,138, lägre än för alla 6 artefaktkategorier. McRae ger samma mönster i artikelns version: djur som helhet 0,111 och alla 541 begrepp 0,034, mot 0,12–0,29 för däggdjur, fågel och fisk. I den rena samvariationsversionen håller det inte: däggdjur (0,089) ligger där under djur som helhet.

Effekten förklaras dock till stor del av två saker:

- **Antalet exemplar.** Ett slumpmässigt urval av 25 djur får C\_X = 0,174, högre än hela domänen med 129 djur (0,145).
- **Avståndstermen.** I artikelns version slår alla 11 verkliga kategorier slumpgrupper av samma storlek, eftersom slumpgrupper per definition ligger vid prototypen.

I den rena samvariationsversionen slår bara fisk och reptil slumpgrupperna tydligt (100:e percentilen). Fordon (93:e) och vapen (90:e) ligger nära. Fågel (13:e), däggdjur (23:e), verktyg (2:a) och kläder (under alla 500 dragningar) är *mindre* samvarierande än slumpvis hopblandade djur respektive artefakter.

### H3: C\_X mot människors uppfattning av kategorin

Artikelns C\_X saknar samband med hur lika människor bedömer att kategorins medlemmar är. Spearman ρ mot bedömd likhet inom kategorin:

| Mått | ρ | p | n |
| --- | --- | --- | --- |
| C\_X, artikelns version | −0,12 | 0,73 | 11 |
| C\_X, ren samvariation, salience-viktad | 0,42 | 0,20 | 11 |
| C\_X, ren samvariation, uniforma vikter | 0,50 | 0,12 | 11 |
| C\_X, ren samvariation, uniforma vikter | 0,52 | 0,048 | 15 |
| C\_X, ren samvariation, kategorimatriser | 0,45 | 0,09 | 15 |
| Familjelikhet (cosinus) | 0,22 | 0,43 | 15 |

Samvariationsversionen har ett svagt samband även när familjelikhet hålls konstant (partiell r = 0,47, p = 0,08, n = 15). Det gäller över alla r (ρ = 0,42–0,56), medan artikelns version ligger mellan −0,21 och 0,01. Med 11–15 kategorier är styrkan låg, så detta är en antydan och inget belagt resultat.

### H4: typikalitet

Artikelns första principalkomponent förutsäger typikalitet väl, men bara för att den pekar mot kategorins mittpunkt. Medelkorrelation med typikalitetsbetyg, med antalet kategorier där sambandet är positivt:

| Prediktor | Medel-ρ | Positiva |
| --- | --- | --- |
| PC1-poäng, universell prototyp (artikeln) | 0,72 | 11/11 |
| Kontrast: projektion på riktningen mot kategorins mitt | 0,72 | 11/11 |
| Familjelikhet (Rosch & Mervis) | 0,69 | 15/15 |
| Bedömd likhet med övriga exemplar | 0,52 | 15/15 |
| Cosinus mot prototypen | 0,47 | 13/15 |
| PC1-anpassning, ren samvariation | 0,08 | 9/15 |
| \|PC1-poäng\|, ren samvariation | 0,06 | 9/15 |

**Figur 1.** Artikelns C\_X mot avståndsandelen, 11 Leuvenkategorier.

&#91;image: C\_X mot avståndsandel\]

**Figur 2.** Bedömd likhet inom kategorin mot C\_X i artikelns version (vänster) och i den rena samvariationsversionen (höger).

&#91;image: C\_X mot människors likhetsbedömningar\]

**Figur 3.** Hur väl olika prediktorer förutsäger typikalitetsbetyg.

&#91;image: Prediktorer för typikalitet\]

**Figur 4.** Ren samvariation i verkliga kategorier jämfört med 500 slumpgrupper av samma storlek från samma domän.

&#91;image: Verkliga kategorier mot slumpkategorier\]

## Tolkning, begränsningar och nästa steg

**Tolkning.** Artikelns val av referenspunkt, den universella prototypen, är det som gör måttet användbart. Det är också det som gör det till något annat än ett samvariationsmått. Med den referenspunkten blir C\_X ett kontrastmått, "hur mycket skiljer sig den här kategorin från allt annat i domänen?", nära besläktat med Roschs cue validity och familjelikhet. Det förklarar typikalitet och överordnade begrepps lägre värden, men inte skillnaden mellan natur och artefakt och inte människors likhetsbedömningar.

Den rena samvariationsdelen, som artikelns argument faktiskt handlar om, ger ett annat mönster. Den gör inte heller skillnad på natur och artefakt, förutsäger inte typikalitet, och flera centrala kategorier (fågel, däggdjur, verktyg, kläder) samvarierar mindre än slumpvis hopblandade grupper. Den enda ljuspunkten är ett måttligt samband med bedömd likhet inom kategorin.

Det stöder invändning 3 från den tidigare genomgången, att referenspunkten gör C\_X till ett särpräglingsmått. Invändning 1, att vikterna smugglar in teori, kan de här data inte avgöra. Människors egna viktighetsbetyg ändrade resultaten lite jämfört med uniforma vikter.

**Begränsningar.**

- Inga direkta koherensbetyg fanns. Bedömd likhet inom kategorin är ett närliggande mått, inte samma sak.
- 11–15 kategorier ger låg statistisk styrka för analyser på kategorinivå.
- Bedömarna var flamländska studenter. Egenskaperna är verbala listor och fångar inte perceptuella dimensioner som storlek och färg på en kontinuerlig skala, vilket konceptuella rum egentligen kräver.
- Den universella prototypen är domänens medelvärde och inte "alla saker". Ett bredare referensurval skulle sannolikt förstärka avståndseffekten.
- Omskalningen av viktighetsbetygen till vikter är ett eget val. Resultaten var dock stabila mellan viktad och oviktad variant.

**Nästa steg.**

- [ ] Samla direkta koherensbetyg, till exempel på Prolific: "hur väl hänger medlemmarna i den här kategorin ihop?" för de 15 Leuvenkategorierna plus ad hoc-kategorier. Det är det avgörande testet.
- [ ] Ta med ad hoc-kategorier ("saker att rädda ur en brand") och sociala kategorier, så att variationen i koherens blir större än mellan 15 vardagskategorier.
- [ ] Testa Lindaanalysen: beräkna C\_X för "bankkassör" och "feministisk bankkassör" i ett egenskapsrum för sociala kategorier.
- [ ] Skriv upp resultaten som en kort replik till Philosophy and the Mind Sciences. Avståndsdekompositionen S\_u = Cov + m mᵀ är ett rent analytiskt bidrag som inte kräver nya data.

## Kod och reproducerbarhet

All kod skickades separat i chatten och körs på under en minut med Python, numpy, pandas, scipy och matplotlib.

| Fil | Vad den gör |
| --- | --- |
| coherence.py | C\_X enligt artikeln, samvariationsvarianten, avståndsandelen och PC-poäng |
| sanity.py | Den syntetiska demonstrationen av origo-artefakten |
| analysis.py | H1–H4, kontroller och McRae-replikationen; skriver results.json och CSV-tabeller |
| figures.py | De fyra figurerna |
| README.md | Var data hämtas och vilka val som gjorts |

Körordning: `python analysis.py <leuven_en_dir> <mcrae.xlsx> out` följt av `python figures.py out`. Slumpfrön är fasta, så alla siffror i den här rapporten återskapas exakt.

## Källor

- Osta-Vélez, M. & Gärdenfors, P. (2026). Conceptual coherence. *Philosophy and the Mind Sciences*, 7. [doi:10.33735/phimisci.2026.12377](https://doi.org/10.33735/phimisci.2026.12377)
- De Deyne, S. m.fl. (2008). Exemplar by feature applicability matrices and other Dutch normative data for semantic concepts. *Behavior Research Methods*, 40, 1030–1048. [doi:10.3758/BRM.40.4.1030](https://www.doi.org/10.3758/BRM.40.4.1030). Rensad kopia: [mikulatomas/dutch-concepts](https://github.com/mikulatomas/dutch-concepts)
- McRae, K. m.fl. (2005). Semantic feature production norms for a large set of living and nonliving things. *Behavior Research Methods*, 37, 547–559. [doi:10.3758/BF03192726](https://www.doi.org/10.3758/BF03192726)
