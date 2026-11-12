# Test av Osta-Vélez & Gärdenfors (2026) "Conceptual coherence" på verkliga normdata

Beräknar artikelns koherensmått C_X och testar dess förutsägelser mot människodata.

## Data (hämtas separat)
- **Leuven Natural Concept Database** (De Deyne m.fl., 2008, *Behavior Research Methods* 40:1030–1048),
  rensad kopia med engelska etiketter: `git clone https://github.com/mikulatomas/dutch-concepts`
  → använd mappen `dutch-concepts/dutch_concepts/data/en`.
  Innehåller exemplar × egenskap-matriser (0–4 bedömare), viktighetsbetyg för egenskaper,
  typikalitetsbetyg och parvisa likhetsbedömningar för 15 kategorier.
- **McRae m.fl. (2005)** egenskapsnormer, `CONCS_FEATS_concstats_brm.xlsx`
  (t.ex. från https://github.com/div5yesh/semantic-feature-extraction).

## Körning
```
pip install numpy pandas scipy matplotlib openpyxl
python sanity.py                                   # syntetisk demonstration
python analysis.py <leuven_en_dir> <mcrae.xlsx> out
python figures.py out
```

## Filer
- `coherence.py` – C_X exakt enligt artikelns avsnitt 3.3 (salience-viktning, universell prototyp,
  geometriska vikter w_i = (1−r)r^(i−1)), plus varianten med kategorins eget medelvärde och
  dekompositionen S_u = Cov + m mᵀ.
- `analysis.py` – hypotes H1–H4, kontroller (storleksmatchning, slumpkategorier, r-robusthet),
  McRae-replikation. Skriver `out/results.json`, `out/leuven_categories.csv`,
  `out/leuven_typicality.csv`, `out/mcrae_categories.csv`.
- `figures.py` – fyra figurer.

## Operationaliseringar (val som kan ifrågasättas)
- Salience = medelvärdet av viktighetsbetyg (−3..3) linjärt omskalat till 0..1.
- Universell prototyp = medelvärdet av alla exemplar i domänen (129 djur resp. 166 artefakter),
  eftersom domänerna har olika egenskapslistor. Ingen gemensam "sak"-prototyp finns i data.
- Huvudspecifikation r = 0,5. Robusthet testad för r ∈ {0,3; 0,5; 0,7; 0,9}.
- Människors koherens operationaliseras som medelvärdet av bedömd parvis likhet inom kategorin.
  Direkta koherensbetyg finns inte i dessa data.

## Embeddinganalys (emb_analysis.py)
Kräver `pip install onnxruntime tokenizers`. Modellerna hämtades som npm-paket (HuggingFace var inte nåbart):
- all-MiniLM-L6-v2, fp32 ONNX: `npm pack @maycombhq/embed-model@2.0.0-beta.1` (mean pooling)
- bge-small-en-v1.5, int8 ONNX: `npm pack impro-plugin-semantic-mute-model` (CLS pooling)
```
python emb_analysis.py <leuven_en_dir> out_emb \
  minilm:<dir>/model.onnx:<dir>/tokenizer.json:mean \
  bge:<dir>/bge-small-en-v1.5-q8.onnx:<dir>/bge-small-en-v1.5-tokenizer.json:cls
python figures_en.py out      # engelska figurer till manuskriptet
```
