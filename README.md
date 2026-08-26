# Interdisciplinary Index <img src="brand/ii-hex.svg" alt="Interdisciplinary Index" align="right" height="139" />

Some research stays inside one field. Some of it draws on several, or gets picked up
by people the author never reads. This measures that, for any researcher in
[OpenAlex](https://openalex.org/), and shows the shape of it.

Four numbers, computed from a researcher's 25 most-cited papers, plus the charts
behind them and a check on whether the profile describes one person at all.

---

## What it measures

Two measures compare **text**, turning each abstract into a vector and taking the
cosine between them. Two compare **fields**, using OpenAlex's subject classification.

### External diversity — how far the work travels

Are the people citing you working on the same thing you are?

```
E = 100 × (1 − mean cosine similarity between a paper and the works citing it)
```

Fifty citing works averaging 0.52 similarity gives 48. Closer citers, averaging
0.70, give 30.

### Internal diversity — how far your own work ranges

Do your own papers resemble each other?

```
I = 100 × mean pairwise cosine distance between your paper abstracts
```

Twenty-five papers means 300 pairs. The similarity heatmap shows the whole matrix;
two dark blocks with a pale gap between them mean two separate strands of work.
This needs no citations, so papers nothing has cited yet still count.

### Reference diversity — how widely you read

Counting fields is not enough: citing Medicine and Nursing is not the same as citing
Medicine and Astronomy, yet a plain count — or Shannon entropy — scores them
identically. This uses **Rao–Stirling diversity**, which folds in how far apart the
fields are:

```
RS = Σ  d_ij · p_i · p_j        over i ≠ j
```

`p_i` is the share of your references in field *i*. `d_ij` comes from OpenAlex's
hierarchy of 26 fields inside 4 domains:

| Relationship | `d_ij` | Example |
|---|---|---|
| Same field | 0 | Medicine – Medicine |
| Different field, same domain | 0.5 | Medicine – Nursing |
| Different domain | 1.0 | Medicine – Physics and Astronomy |

So two reference lists that both split 50/50 and have identical entropy still
separate: Medicine + Nursing scores 25, Medicine + Astronomy scores 50.

Reported alongside it is the **effective number of fields**, `exp(H)` — the number
of evenly-used fields that would produce the same spread.

### Bridge — who reads you that you don't read

How much of your audience does your own reading fail to explain?

```
B = 100 × Σ  max(0, aᶠ − sᶠ)
```

`aᶠ` is field *f*'s share of the works citing you; `sᶠ` its share of your references.
Only the excess counts, so a field you cite more than it cites you contributes zero.

It is weighted by volume: a field supplying 1.4% of your references but 14.6% of your
citers contributes 13 points, while a single stray citation out of 500 moves the
score by two-tenths. And it is continuous — no threshold for a field to fall the
wrong side of.

### The composite

The unweighted mean of the four. Equal weights are a **choice**, not a neutral
default, and the four are not independent — external and internal diversity share an
embedding space, reference diversity and bridge share a field taxonomy. Prefer the
four-part profile; read the single number as a rough summary.

---

## Reading the numbers

**There is no absolute scale, and the interface shows no grades.** Any two scientific
abstracts share a great deal of ordinary English, so the cosine-based measures rarely
approach zero even for tightly focused work. A 50 does not mean "half as
interdisciplinary as possible".

These numbers earn their keep by comparison: one researcher against another analysed
the same way, or the same researcher at two points in a career.

---

## Is the profile one person?

OpenAlex assigns author identifiers algorithmically, so one record can hold several
people's work. This is the dangerous case: unrelated papers read as range, so **all
four measures rise at once** and the score peaks exactly when the data are worst.

Before analysing, one request samples the record and checks co-author cohesion,
institutional spread, byline names, career span, conflicting ORCIDs, and whether two
field communities share any collaborators. Co-author cohesion is what separates real
breadth from a name collision: a genuine polymath keeps a recurring core of
collaborators across fields; two people who share a name do not.

Papers can also be excluded by hand — OpenAlex files the occasional stray paper under
a record, and one misattributed paper affects every measure at once.

---

## Using it

**Analyse** — search by name, ORCID, or OpenAlex ID, pick the right profile from the
candidate list, and run. About ten seconds.

**Compare** — up to three researchers on shared axes, with a written read of where
they differ, how much their reading and their audiences overlap, and whether they
bridge into the same places or different ones.

**Export** — a self-contained HTML report with every chart.

Citing works are drawn with a seeded random sample rather than "most recent", so a
rerun reproduces the result. Self-citations are excluded.

---

## Running it

Requires Python 3.10 or newer.

```bash
git clone https://github.com/alkat19/interdisciplinary-index-analyzer.git
cd interdisciplinary-index-analyzer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python interdisciplinary_app.py
```

Then open <http://localhost:7860>. The first launch downloads the embedding model.

### OpenAlex API key

OpenAlex meters usage as a daily allowance rather than a request count. Anonymous
access gets $0.10 a day, roughly 19 analyses; a **free** key raises it to $1, roughly
190. Neither tier needs a payment method, so running out returns an error, never a
charge.

```bash
export OPENALEX_API_KEY='your-key-here'
```

It is sent as an `Authorization: Bearer` header, and the app reports at startup
whether it was accepted.

---

## Limits

- Only the 25 most-cited papers are analysed, so internal and reference diversity
  describe a researcher's most-cited output rather than everything they have written.
- External diversity and bridge need papers that have been cited.
- Field distance uses a four-domain hierarchy, which is coarse: it cannot tell that
  two fields inside one domain are further apart than two others.
- Embedding distance partly reflects writing style and venue conventions, not only
  subject matter.
- Coverage is reported, not hidden — works OpenAlex has not classified are counted
  and shown as a percentage beneath the profile.

---

## Built with

[OpenAlex](https://openalex.org/) · [Gradio](https://gradio.app/) ·
[Sentence Transformers](https://www.sbert.net/) with
[`minishlab/potion-base-32M`](https://huggingface.co/minishlab/potion-base-32M) ·
[Plotly](https://plotly.com/) · [KeyBERT](https://github.com/MaartenGr/KeyBERT)
