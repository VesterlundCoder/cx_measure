# Material provenance and verification

Repository assembled on 2026-11-12.

## Original inputs

| Original input | Repository copy |
| --- | --- |
| `cx_coherence_test.zip` | `materials/cx_coherence_test.zip`; expanded in `sbert/upstream/` |
| `cx_coherence_test-2.zip` | `materials/cx_coherence_test-2.zip`; expanded in `norms/` |
| `What does C_X measure An empirical test of the PCA account of conceptual coheren.md` | `materials/manuscript_original.md` |
| `Test av C_X – konceptuell koherens på verkliga normdata.md` | `sbert/reference/normdata_report.md` |
| `Osta-Velez-Gärdenfors-PhiMiSci-2026.pdf` | `sbert/reference/article.pdf`; extracted text in `article.txt` |
| Additional local MPNet project | `sbert/`, including both runs' predictions and embedding caches |

The ZIP archives and `materials/manuscript_original.md` are unchanged.
`manuscript.md` is a reading copy: the repository placeholder is replaced
with the live URL, figure placeholders are replaced with relative image
links, and the equation uses a GitHub math block. Its scientific claims
remain those of the supplied manuscript.

The expanded archive scripts and result tables are unchanged. Figures in
`norms/` were generated from the supplied tables using the supplied plotting
scripts; regenerating figures does not independently verify the underlying
analyses.

## What was run and checked

- The additional MPNet analysis was run on both bare labels and
  `This is {label}.`, covering 15 categories and 5,723 within-category pairs.
- Central Leuven feature-norm coherence, similarity and typicality
  baselines were recomputed and checked against the original CSV tables.
  The largest absolute deviation was `5.08e-13`.
- Independent numerical checks covered the covariance decomposition, the
  Mantel permutation implementation, pair identity/alignment and the
  recorded environment lock.
- During repository assembly, the supplied synthetic demonstration and
  both sets of norm-data plotting scripts were executed.
- The McRae replication and supplied MiniLM/BGE model runs were **not**
  independently repeated. Their code and result summaries are preserved
  with the original manuscript.

The original target article is theoretical and does not supply this
repository's empirical results.

## Differences that matter when comparing pipelines

1. **Representation and input.** The supplied embedding analysis uses
   MiniLM/BGE with bare names and `a {name}`. The additional MPNet analysis
   uses bare names and `This is {name}.`. These are separate experiments.
2. **Universal prototype.** In `norms/emb_analysis.py`, `allE` stacks
   category memberships, so its mean weights repeated labels more than
   once: 418 memberships rather than 415 distinct labels. The manuscript
   describes a mean over 415 exemplars. The MPNet global mean uses 415
   unique label vectors. Neither finite sample is a literal prototype of
   all things. The supplied implementation and manuscript are preserved
   so this difference remains inspectable.
3. **Pairwise feature baseline.** The supplied ONNX analysis uses each
   category's feature matrix. The MPNet analysis uses the common domain
   matrix for the 11 animal/artifact categories, matching the supplied
   main norm-data analysis. This explains why the reported mean feature
   cosine correlation is approximately 0.65 in one comparison and 0.64
   in the other.
4. **Prototype cosine.** The manuscript's approximately 0.47 typicality
   result corresponds to the supplied *unweighted* feature-prototype
   cosine. The MPNet report's approximately 0.42 feature-prototype
   baseline is salience-weighted.
5. **Model artifact provenance.** The original ONNX README identifies an
   exact MiniLM npm package version, but no BGE package version or model
   file checksums. The MPNet analysis pins a Hugging Face model revision
   and includes its vectors and exact text inputs.

These notes document the material as supplied and the additional analysis;
they do not revise the manuscript's argument or establish that every claim
in it has been independently verified.
