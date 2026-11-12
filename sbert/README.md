# C_X och Sentence-BERT på begreppsnormer

Ett lokalt projekt i beräkningsfilosofi: jämför cosinuslikhet från en fryst
Sentence-BERT-modell med människors likhets- och typikalitetsbedömningar,
och med operationaliseringen av Osta-Vélez & Gärdenfors (2026) i det bifogade
normdatatestet.

Grundartikeln är teoretisk och innehåller ingen numerisk Leuven-benchmark.
`upstream/` är den oförändrade zipbilagan, inte kod eller empiriska resultat
publicerade av artikelns författare. `reference/` innehåller artikeln och
den bifogade rapporten.

## Modell och data

- Modell: `sentence-transformers/all-mpnet-base-v2`, 768 dimensioner,
  revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, Apache 2.0.
- Leuven: https://github.com/mikulatomas/dutch-concepts,
  commit `5a519d4f63318dd8bc0322b8cc00a31c30ef5146`.
- 15 kategorier. Artikelns domänrefererade C_X går att återskapa för de
  11 kategorier som har gemensamma egenskapsmatriser för djur/artefakter.
- Engelska etiketter används exakt som i databasen, inklusive eventuella
  parenteser och nederländska särskiljningar. Ingen kategorietikett läggs till.
  `inputs.csv` redovisar varje exakt modellindata.
- Primär analys: exemplarens namn. Känslighetsanalys: `This is {label}.`.
  Ingen träning, promptoptimering eller val av modell utifrån testresultaten.
- McRae-tabellerna finns bevarade i bilagan men analyseras inte på nytt här;
  Leuven har människobedömningarna som behövs för denna jämförelse.

## Körning

Kör kommandona från denna mapp (`cd sbert` från repots rot).
Repots övergripande README visar alternativet med en gemensam miljö i roten.

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock
git clone https://github.com/mikulatomas/dutch-concepts data/dutch-concepts
git -C data/dutch-concepts checkout 5a519d4f63318dd8bc0322b8cc00a31c30ef5146
TOKENIZERS_PARALLELISM=false .venv/bin/python compare.py
TOKENIZERS_PARALLELISM=false .venv/bin/python compare.py --condition sentence
.venv/bin/python report.py
```

Modellen laddas automatiskt ned till Hugging Faces lokala cache första gången.
Embeddings sparas under `data/` med en nyckel som inkluderar modellrevision och
alla indatatexter. CPU används för reproducerbarhet på denna Mac.

## Mått och analysval

1. **Kategorilikhet:** medelcosinus över unika par av olika exemplar.
   Spearmankorrelation mot medelvärdet av människors parvisa likhet,
   både för samma 11 kategorier som C_X och för samtliga 15.
2. **Parvis likhet:** Spearman inom varje kategori. Tvåsidigt Manteltest
   med 1 999 permutationer av exemplaridentiteter; Holmkorrektion över 15
   kategorier. Par med gemensamma exemplar behandlas inte som oberoende.
3. **Typikalitet:** varje exemplars medelcosinus till övriga exemplar och
   cosinus till en prototyp där exemplaret självt har lämnats utanför.
   Samband beräknas separat per kategori. Fisher-z-medel och aritmetiskt
   medel redovisas separat; bilagans ”medel-rho” är Fisher-z-medel.
4. **C_X:** bilagans implementation återanvänds, r=0,5. Nyberäknade
   egenskapsbaslinjer kontrolleras numeriskt mot de bifogade CSV-filerna.
   C_X på embeddings är en separat explorativ analys med uniforma
   dimensionsvikter och domänmedel/globalt medel/egen medelpunkt som referens.
   Embeddingdimensioner har inte uppmätta psykologiska salience-vikter.
5. **Natur–artefakt:** exakt tvåsidigt permutationstest över 462 indelningar
   av fem djur- och sex artefaktkategorier. **Över-/underordning:** deskriptiv
   jämförelse av domänens medelcosinus med underkategoriernas.
6. **Skillnader mellan modeller:** parad bootstrap över kategorier, 3 000
   dragningar, för skillnaden i kategorikorrelation och i typikalitetskorrelation.
   Kategorier är ett litet, utvalt urval; intervallen är explorativa.

Slumpfrö: 20261112. Diagonaler exkluderas. Ett kategorimått C_X ger inte en
unik parvis prediktion; egenskapsbaserad cosinus används därför som parvis
jämförelsemodell. PC1:s tecken orienteras utan tillgång till typikalitetsbetygen.
Även absolut PC1-poäng och kvadratisk axelanpassning redovisas, eftersom
artikelns typikalitetsdiskussion omfattar mer än signerade poäng.

## Utdata

`REPORT.md` sammanfattar de faktiska körningarna. Varje indataformulering har
en egen mapp under `results/` med sammanfattning, kategoritabell,
typikalitetskorrelationer, exemplarpoäng, samtliga par, modellindata,
replikationsavvikelser och figurer.

## Tolkningsgränser

Människors likhetsbetyg är inte direkta koherensbetyg. Engelska
textrepresentationer jämförs med normer från nederländsktalande deltagare;
översättning och flertydiga ord kan påverka resultaten. En språkmodell kan
ha mött relaterade normdata under träning; detta har inte kontrollerats.
Inga resultat här visar att modellen använder samma kognitiva mekanism som
människor. Skillnader mellan egenskapsbaserat C_X och embeddingcosinus
ändrar både representation och mått.
