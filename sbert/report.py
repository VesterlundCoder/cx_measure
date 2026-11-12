"""Write a Swedish report from completed comparison runs, without fitting anything."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def f(value):
    return f"{value:.3f}"


def interval(values):
    return f"[{f(values[0])}, {f(values[1])}]"


def mdtable(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    base = ROOT / "results"
    a = json.loads((base / "labels/summary.json").read_text())
    b = json.loads((base / "sentence/summary.json").read_text())
    cats = pd.read_csv(base / "labels/categories.csv")
    typ = pd.read_csv(base / "labels/typicality.csv")
    delta = a["category_delta_sbert_minus_cx_11"]
    td = a["typicality_delta_sbert_minus_PC1_11"]
    category_names = {
        "sbert_cosine": "SBERT: medelcosinus",
        "feature_cosine": "Egenskaper: medelcosinus",
        "cx_features_universal": "C_X: universell domänprototyp, salience",
        "cx_features_own": "C_X: egen medelpunkt, salience, domänmatris",
        "cx_features_own_category": "C_X: egen medelpunkt, salience, kategorimatris",
        "cx_features_own_uniform": "C_X: egen medelpunkt, uniforma vikter",
        "cx_sbert_domain": "C_X på SBERT: domänprototyp",
        "cx_sbert_global": "C_X på SBERT: global prototyp",
        "cx_sbert_own": "C_X på SBERT: egen medelpunkt",
    }
    typ_names = {
        "sbert_mean_cosine": "SBERT: medelcosinus till övriga",
        "sbert_prototype_loo": "SBERT: cosinus till prototyp utan exemplaret",
        "feature_PC1_signed": "Egenskaper: signerad PC1, universell prototyp",
        "feature_PC1_absolute": "Egenskaper: absolut PC1, universell prototyp",
        "feature_PC1_alignment": "Egenskaper: kvadratisk PC1-anpassning",
        "feature_contrast": "Egenskaper: kontrastprojektion",
        "feature_family_resemblance": "Egenskaper: familjelikhet",
        "feature_prototype_cosine": "Egenskaper: prototypcosinus",
        "human_similarity_to_others": "Människors medellikhet till övriga",
    }
    crows = [[scope, category_names[m], f(s["rho"]), f(s["p"])]
             for scope in ["11", "15"] for m, s in a["category_level"][scope].items()]
    trows = [[scope, typ_names[m], f(s["fisher_mean_rho"]), f(s["arithmetic_mean_rho"]),
              f'{s["positive"]}/{s["n"]}']
             for scope in ["11", "15"] for m, s in a["typicality"][scope].items()]
    hrows = [[category_names.get(m, {"human_similarity": "Människors likhet"}.get(m, m)),
              f(s["animal"]), f(s["artifact"]), f(s["p_exact"])] for m, s in a["H1"].items()]
    prows = [[r.category, int(r.n), f(r.pairwise_rho_sbert), f(r.pairwise_p_mantel),
              f(r.pairwise_p_holm), f(r.pairwise_rho_features)]
             for r in cats.itertuples()]
    sensitivity = []
    for scope in ["11", "15"]:
        for condition, s in [("Namn", a), ("This is …", b)]:
            c = s["category_level"][scope]["sbert_cosine"]
            t = s["typicality"][scope]["sbert_mean_cosine"]
            sensitivity.append([condition, scope, f(c["rho"]), f(c["p"]), f(t["fisher_mean_rho"])])
    percat = []
    for r in cats.itertuples():
        t = typ.set_index("category").loc[r.category]
        cx = getattr(r, "cx_features_universal")
        percat.append([r.category, int(r.n), f(r.human_similarity), f(r.sbert_cosine),
                       "—" if pd.isna(cx) else f(cx), f(t.sbert_mean_cosine),
                       "—" if pd.isna(t.feature_PC1_signed) else f(t.feature_PC1_signed)])
    h2rows = []
    for domain, s in a["H2"].items():
        values = list(s["subordinate_cosines"].values())
        h2rows.append([domain, s["n"], f(s["superordinate_cosine"]),
                       f(min(values)), f(max(values)),
                       f'{sum(v > s["superordinate_cosine"] for v in values)}/{len(values)}'])
    text = f"""# Sentence-BERT och C_X på verkliga normdata

Kördatum: 2026-11-12. Modell: `{a['model']}`, revision `{a['revision']}`.

## Vad som faktiskt jämförs

Grundartikeln, Osta-Vélez & Gärdenfors (2026), *Conceptual coherence*,
definierar C_X men redovisar inga numeriska Leuven-resultat. Referensresultaten
kommer från den bifogade normdataanalysen, inte från författarnas egen
empiriska utvärdering. Här har dess centrala Leuven-baslinjer räknats om från
rådata och verifierats mot bilagans tabeller: största absoluta avvikelse
{a['baseline_max_absolute_error']:.2e}.

Testet omfattar {len(cats)} kategorier, {a['category_memberships']} kategorimedlemskap,
{a['unique_labels']} unika modellindata inklusive hela domänernas exemplar och
{a['pairs']} unika par inom kategorierna. Modellens engelska namnrepresentationer
jämförs med människors likhets- och typikalitetsnormer.
Samma frysta modell används genomgående utan anpassning till normerna.

## Resultat på kategorinivå

På samma 11 kategorier är sambandet med människors bedömda likhet
ρ = {f(a['category_level']['11']['sbert_cosine']['rho'])} för SBERT-cosinus och
ρ = {f(a['category_level']['11']['cx_features_universal']['rho'])} för C_X med
universell domänprototyp på egenskapsnormerna.
Skillnaden är Δρ = {f(delta['delta_rho'])}, med explorativt parat
bootstrapintervall 95 % {interval(delta['ci95'])}.
Detta intervall avser skillnaden mellan två korrelationer på samma kategorier.

{mdtable(["Kategorier", "Mått", "Spearman ρ", "p (okorrigerat)"], crows)}

C_X på SBERT är en separat explorativ tillämpning. Den använder uniforma
dimensionsvikter eftersom det inte finns uppmätta salience-vikter för
modellens 768 dimensioner. Global prototyp är medelvärdet av alla unika
etikettvektorer i detta begränsade urval, inte en verklig prototyp för ”sak”.
Domänrefererat C_X på SBERT korrelerar med sin avståndsandel med
ρ = {f(a['cx_sbert_vs_distance_share']['rho'])}.
Den direkta rangkorrelationen mellan SBERT-medelcosinus och egenskapsbaserat
C_X med universell domänprototyp är
ρ = {f(a['sbert_cosine_vs_cx_features_11']['rho'])},
p = {f(a['sbert_cosine_vs_cx_features_11']['p'])}, n = 11.

![Kategorilikhet](results/labels/category_similarity.png)

## Typikalitet

På samma 11 kategorier är Fisher-z-medel av korrelationen mot typikalitet
{f(a['typicality']['11']['sbert_mean_cosine']['fisher_mean_rho'])} för SBERT-medelcosinus
och {f(a['typicality']['11']['feature_PC1_signed']['fisher_mean_rho'])} för
bilagans signerade PC1-prediktor.
SBERT är bättre i {td['sbert_better']} av {td['n']} kategorier.
Medelvärdet av de parade skillnaderna i rå ρ är {f(td['mean_delta_rho'])},
bootstrapintervall 95 % {interval(td['ci95'])}.

{mdtable(["Kategorier", "Prediktor", "Fisher-z-medel ρ", "Aritmetiskt medel ρ", "Positiva"], trows)}

I detta material ger medelcosinus till övriga och cosinus till prototypen
utan exemplaret samma rangordning, så deras Spearmankorrelationer
sammanfaller. De är närbesläktade mått, inte två oberoende bevis.
Absolut PC1-poäng och kvadratisk axelanpassning visas också, eftersom
artikelns typikalitetsdiskussion inte enbart handlar om signerade PC1-poäng.

![Typikalitet per kategori](results/labels/typicality.png)

## Parvis likhet

SBERT:s Fisher-z-medel av de 15 inomkategorikorrelationerna är
{f(a['pairwise']['sbert_fisher_mean_rho'])}; motsvarande egenskapsbaserade
cosinus ger {f(a['pairwise']['features_fisher_mean_rho'])}.
SBERT har positiv korrelation i {a['pairwise']['sbert_positive']}/15 kategorier,
och {a['pairwise']['sbert_holm_significant']}/15 är signifikanta efter
Holmkorrektion av tvåsidiga Manteltest.

{mdtable(["Kategori", "n", "SBERT ρ", "Mantel-p", "Holm-p", "Egenskapscosinus ρ"], prows)}

Varje Manteltest använder {a['permutations']} permutationer av exemplaridentiteter.
Det bevarar beroendet mellan par med gemensamma exemplar. Vanliga p-värden
med varje par som en oberoende observation skulle vara missvisande.
C_X är ett kategorimått och ger inte i sig en unik prediktion för varje par.
Parjämförelsen använder därför egenskapscosinus som baslinje.

## Natur–artefakt och överordnade kategorier

Exakt tvåsidigt permutationstest över 462 indelningar, fem djurkategorier
och sex artefaktkategorier:

{mdtable(["Mått", "Djur", "Artefakter", "p"], hrows)}

Deskriptiv jämförelse för SBERT-cosinus:

{mdtable(["Domän", "n", "Hela domänen", "Lägsta underkategori", "Högsta underkategori", "Underkategorier över domänen"], h2rows)}

Detta mönster för cosinus är ett resultat om likhetsbaserad sammanhållning,
inte en direkt testning av C_X:s kompressionsmekanism.

## Känslighet för indataformulering

Samma modell och exemplar, med engelska namn respektive den i förväg valda
mallen `This is {{label}}.`. Kategorinamnet läggs inte till.

{mdtable(["Indata", "Kategorier", "ρ kategorilikhet", "p", "Fisher-z-medel ρ typikalitet"], sensitivity)}

Mallen ger parvis Fisher-z-medel ρ =
{f(b['pairwise']['sbert_fisher_mean_rho'])} och natur–artefakt-p =
{f(b['H1']['sbert_cosine']['p_exact'])}.
För mallen är skillnaden i kategorikorrelation mot C_X på samma 11 kategorier
Δρ = {f(b['category_delta_sbert_minus_cx_11']['delta_rho'])}, med parat
bootstrapintervall 95 %
{interval(b['category_delta_sbert_minus_cx_11']['ci95'])}.
De två formuleringarna är känslighetsanalyser av samma modell.
Ingen formulering har valts i efterhand som ”bäst”.

## Resultat per kategori, primär analys

{mdtable(["Kategori", "n", "Människolikhet", "SBERT-cosinus", "C_X normdata", "SBERT typikalitets-ρ", "PC1 typikalitets-ρ"], percat)}

## Hur resultaten bör tolkas

Likhet inom en kategori, typikalitet hos dess medlemmar och konceptuell
koherens är skilda mål. Ett bättre samband med likhetsbetyg räcker inte för
att etablera en bättre generell teori om koherens. Inga direkta koherensbetyg
finns i denna analys.

Jämförelsen mellan egenskapsbaserat C_X och SBERT-cosinus ändrar både
representation och mått. Egenskapscosinus och C_X på embeddings ger
kompletterande jämförelser, men isolerar inte alla metodskillnader.
11–15 utvalda kategorier ger osäkra korrelationer och bootstrapintervall.
Övriga tabellers p-värden är explorativa och okorrigerade, utom uttryckligen
Holm-korrigerade parvisa test.

Grundartikeln säger uttryckligen i avsnitt 3.3 att referensen till den
universella prototypen ska göra måttet användbart även när direkt samvariation
är låg. Att avståndstermen bidrar är därför inte i sig ett programmeringsfel
eller ett resultat som författarna uttryckligen utesluter. Det är däremot
centralt för vilken tolkning man kan ge måttet.

Deltagarna var nederländsktalande medan modellen får engelska etiketter.
Databasens etiketter bevaras exakt, inklusive flertydigheter och eventuella
nederländska parenteser. Träningsöverlapp med normdata är inte kontrollerat.
Resultaten är en jämförande pilotstudie utan träning på testbetygen, inte
ett bevis för kognitiv mekanism eller generell modellöverlägsenhet.

## Reproducerbarhet och filer

- `compare.py`: inläsning, SBERT, baslinjereplikation, statistik och figurer.
- `results/labels/` och `results/sentence/`: alla sammanfattningar och råa
  prediktioner, inklusive samtliga par, exemplarpoäng och exakta modellindata.
- `data/embeddings_*.npz`: cachade vektorer.
- `requirements.lock`: de installerade paketversionerna.
- `README.md`: körkommandon, analysval och datakällor.

Leuven-commit: `{a['leuven_commit']}`.
Python {a['python']}, enhet `{a['device']}`, slumpfrö {a['seed']}.
McRae ingår endast som bevarat referensmaterial i zipbilagan; ingen ny
McRae-replikation påstås här.
"""
    (ROOT / "REPORT.md").write_text(text)
    print(ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
