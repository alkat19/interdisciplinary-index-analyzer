# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Gradio app that scores a researcher's interdisciplinarity from OpenAlex data. Two
modules, no framework beyond Gradio:

- **`reproducible_cache.py`** — the OpenAlex data layer. Fetching, retry, typed
  errors, and on-disk caching. Knows nothing about metrics.
- **`interdisciplinary_app.py`** — everything else: metrics, coherence checking,
  7 Plotly charts, HTML export, inline CSS, and the Gradio UI.

## Running it

Requires **Python 3.10+** (the code uses `dict | None` annotations).

```bash
OPENALEX_API_KEY='...' .venv/bin/python interdisciplinary_app.py
```

Serves on `:7860`. The key is read from the environment only and sent as an
`Authorization: Bearer` header — never put it in source, the repo is public.
Startup makes one free single-entity request and logs whether the key was
accepted; `budget $1.00` means yes, `$0.10` means it was ignored.

There is no test suite. Verification in this project has been: run it, drive it in
a browser, and check invariants with the AST audit below.

## The four measures

All computed over the **25 most-cited papers** with an abstract and a DOI.

| Measure | Formula |
|---|---|
| External diversity | `100 × (1 − mean cosine similarity to citing abstracts)` |
| Internal diversity | `100 × mean pairwise cosine distance between own abstracts` |
| Reference diversity | Rao–Stirling `Σ d_ij·p_i·p_j` over reference field shares |
| Bridge | `100 × Σ max(0, audience_share − reference_share)` |

Composite is their unweighted mean. `d_ij` comes from OpenAlex's hierarchy —
same field 0, same domain 0.5, different domain 1.0 — with the domain learned
from cached topic records, not hardcoded.

## Decisions that look like bugs but aren't

Each of these was tried the other way and reverted. Don't undo without reading why.

- **No category labels ("High", "Very High").** Removed deliberately. Cosine
  similarity between scientific abstracts rarely falls below ~0.25, so the low
  end of the scale is unreachable and a grade would overstate what the number means.
- **Disparity from the taxonomy, not embeddings.** Embedding field *names* put
  every pair in a 0.24–0.43 band (`d(Medicine, Nursing)` 0.30 vs
  `d(Medicine, Astronomy)` 0.33), so Rao–Stirling collapsed onto its balance term.
- **Bridge is positive imbalance, not a set difference or a threshold.** The set
  version let one stray citation count as much as four hundred; a 1%-of-references
  threshold created a cliff that excluded a field supplying 14.6% of the audience.
- **Only most-cited selection.** A career-wide random sample existed and was
  removed: a third of a random draw has no citations, so External Diversity and
  Bridge silently fell back to a cited subset anyway.
- **Internal diversity includes uncited papers.** It needs no citing work;
  restricting it would reintroduce citation bias.
- **Plotly stays.** Gradio's native plots can't do the diverging flow chart or a
  scaling heatmap, and the HTML export depends on `fig.to_html()`.
- **Results tables are hand-rendered HTML, not `gr.Dataframe`.** Gradio truncates
  headers instead of wrapping them.
- **The author picker is a `gr.Radio` of two-line cards, not a table.** A table
  truncates every column in a narrow pane, destroying the disambiguating detail.

## Gradio 6 gotchas

Version 6.26 broke several things that v4 patterns rely on:

- `css`, `theme`, `js`, `head` belong on **`launch()`**, not the `Blocks`
  constructor — they are silently swallowed there.
- **`@import` in injected CSS is ignored** (constructed stylesheets). Fonts are
  linked via `launch(head=PAGE_HEAD)`.
- `show_progress` decorates **every visible output component**, which produced two
  progress bars on one event. The long-running handlers are now async generators
  that yield their own `progress_block()` and pass `show_progress="hidden"`.
- **CPU work must go through `asyncio.to_thread`.** Embedding ran on the event
  loop and froze the whole server for the length of an analysis.
- **The same `gr.update()` object cannot be delivered to two outputs.** Assigning
  one dict to two slots silently updates only the first; build a separate dict per
  slot. This is why the progress bar has two status slots, one at each end of the
  page — the results page is ~10,000px tall and the recalculate control sits at the
  bottom, far below the main one.
- **`launch(favicon_path=...)` emits no icon link at all.** The tab icon is set
  instead by a data-URI `<link rel="icon">` inside `PAGE_HEAD`, built from
  `MARK_SVG`, which needs no static route and cannot 404.
- `<` inside a LaTeX block breaks the markdown renderer — use `\lt`.
  `gr.Markdown` needs `latex_delimiters` passed explicitly; the default is `None`.

## OpenAlex specifics

- Usage is metered as a **daily budget**, not a request count: $0.10/day anonymous,
  $1/day with a free key. Single-entity fetches are free, list+filter is $0.10 per
  1,000, **search is $1 per 1,000 — ten times a data call**. Neither tier needs a
  payment method, so exhausting it returns 429, never a charge.
- One analysis is ~42 list calls, so roughly 190 a day on a free key.
- Citing works use `sample=N&seed=` (reproducible) rather than
  `sort=publication_date:desc`, which over-represents whatever is currently fashionable.
- Self-citations are excluded with `author.id:!{id}` — verified working.
- A rejected key returns **401 on every request**, not a downgrade to anonymous.
- `topics.field.id` is not a valid `group_by` on `/authors`; classify fields from
  the works instead.

## Caching

`cache_dir` is a fresh temp directory per Gradio session. The cache stores **raw
fetched data only — never computed metrics**, so changing a formula needs no
refetch. The fetch parameters *are* the cache key, so changing `top_n`, `seed` or
`exclude_self_citations` invalidates it automatically. `CACHE_SCHEMA` guards
shape changes.

## Verifying a change

`py_compile` misses a lot here — it passed on an orphaned code block left by a bad
regex, and on handler/output arity mismatches. Run this instead:

```bash
python3 -m py_compile interdisciplinary_app.py reproducible_cache.py
```

then an AST pass for undefined names and unused imports, plus a check that every
`.click()`/`.change()` handler's parameter count matches its `inputs` list and its
return arity matches its `outputs` list. Arity drift is the most common breakage
when rewiring the UI, and it fails silently at runtime.

When editing by script, **assert each replacement individually** — a single
`assert s != o` at the end hides edits that silently didn't match.

## Two features with rules of their own

**Excluding a paper.** OpenAlex files stray papers under a record, and one
misattributed paper contaminates all four measures — its text, its references and
its audience all count. The exclusion list is built *before* the filter runs, so
an excluded paper stays visible and ticked and can be put back individually.
Recalculating reuses the cache, so it costs no requests and takes ~0.3s.

**Comparison insights.** `comparison_insights()` narrates only gaps above
`MEANINGFUL_GAP` (8 points) — two 25-paper samples differ by a few points for no
reason, and the app reports point estimates with no intervals, so smaller gaps are
simply not narrated. It deliberately emits no ranking or verdict: the measures
describe citation patterns, not quality. Overlap uses the summed minimum of the
two field-share distributions, which reads as "the fraction of one you could lay
on top of the other".

## The mark

`MARK_SVG` is inlined in `interdisciplinary_app.py` rather than served: under 1KB,
so no static route, no extra request, nothing to 404. It appears in the app
masthead, in the exported report's masthead, and as the favicon. Its clip id is
namespaced (`iiHexClip`) because the export page embeds the same markup twice.

`brand/ii-hex.svg` is the source of truth — **keep the two in step**, there is no
build step that syncs them. `brand/README.md` covers the cuts and the geometry.

## Known limits

Documented in the About tab; worth not rediscovering:

- Only the 25 most-cited papers, so Internal and Reference diversity describe a
  researcher's most-cited output, not everything they wrote.
- Four-domain disparity is coarse — it cannot tell that Mathematics and Physics
  are closer than Mathematics and Materials Science.
- Embedding distance partly reflects writing style, not only subject.
- The coherence check is a heuristic on a bounded sample: a split confined to a
  small corner of a very large record can slip through.

## Not done

`cleanup_session_cache` is defined and wired only as an age-gated startup sweep;
Gradio's `unload` hook takes no arguments so it cannot target the session's own
directory. Per-metric drill-down was considered and skipped — the seven charts
already serve that purpose.
