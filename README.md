# Bile Acid Analysis Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21513592.svg)](https://doi.org/10.5281/zenodo.21513592)

**Version 4.7.0** · [Web application](https://bile-acid-pipeline.streamlit.app/)

A comprehensive, user-friendly pipeline for analyzing LC-MS bile acid data with automated statistical testing and publication-quality visualizations.

## Features

- **Automatic data structure detection** — recognizes 44 bile acid species from LC-MS output and detects the sample ID, group, and study factors
- **One-, two-, and three-way designs** — group comparisons, factorial ANOVA, and interaction plots
- **Assumption-aware statistics** — checks normality and homoscedasticity before selecting parametric or non-parametric tests
- **Cross-analyte FDR control** — Benjamini-Hochberg q-values are computed by default within each analysis section and scale; q-significant omnibus tests gate post-hoc comparisons
- **LOD-aware aggregate testing** — totals are screened by their LOD-bound replacement burden, while `total_all` is always retained with a warning
- **Comprehensive calculations** — group totals, pool percentages, and clinical ratios
- **User-defined group order** — control the display order used in figures, reports, and exports without changing statistics
- **Bounded-memory workflow** — only the active analysis section is rendered, statistics are computed lazily, and export figures are generated one at a time
- **Exact-style figure exports** — every figure viewed in the app is recreated in the ZIP with the same colors, plot type, selections, log scale, group order, and tab-specific options
- **Complete export package** — Excel data workbooks, a statistical report, and every figure type, including tabs that were never opened

## Installation

```bash
# Clone or download the pipeline
cd bile_acid_pipeline

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application

https://bile-acid-pipeline.streamlit.app/

### Local Hosting

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Input Format

Upload an Excel (`.xlsx`, `.xls`) or ODS (`.ods`) workbook containing:

- **LC-MS standards sheet** — the standard-curve data used for automatic LOD detection
- **Sample sheet** — one row per sample, with sample IDs, groups, and bile acid concentrations
- **Optional metadata sheet** — additional sample-level factors

The sample sheet should contain:

- **Rows:** samples
- **Columns:** bile acid species matching the panel names
- **First column(s):** sample ID and group/type
- **Values:** concentrations in a consistent unit across samples; the sidebar units label is used for display
- **Below LOD:** values such as `-----`, `LOD`, `BLQ`, or `ND`

Example:

| Type | Sample_ID | TCA | GCA | TCDCA | GCDCA | ... |
|------|-----------|-----|-----|-------|-------|-----|
| HD-1 | 81-0210 | 37.62 | 194.1 | 231.9 | 3220.44 | ... |
| HD-2 | 81-0211 | 66.66 | 287.58 | 158.46 | 1534.02 | ... |
| AC-1 | 60-677 | 45.2 | ----- | 189.3 | 2890.1 | ... |

For multi-factor designs, add one `Factor_` column per independent variable:

| Factor_Age | Factor_Sex | Sample_ID | CA | TCA | GCA | ... |
|------------|------------|-----------|----|-----|-----|-----|
| Aged | Male | S001 | 125.4 | 142.8 | 98.6 | ... |
| Young | Female | S002 | 98.6 | 312.5 | 287.9 | ... |

The app also accepts factor columns supplied through a metadata sheet.

## Exports

The **Export** section provides:

- Individual Excel workbooks for concentrations, totals, ratios, and percentages
- A LOD-highlighted workbook showing every replaced value
- A complete statistical report in Excel
- A ZIP package containing the data workbooks, the statistical report, and PDF figures

The ZIP is generated on demand and written to disk. Figures are rebuilt from lightweight recipes one at a time, so the app does not keep every figure in memory. Viewed figures take precedence over the standard export variants with the same name; the standard variants and always-on defaults cover figure types from tabs that were not visited.

## Group Order

The **Group order for figures** control sets the left-to-right order of groups and factor levels in figures, report figures, and ZIP exports. Changing the order does not recompute or alter any statistics. The order is stored as display metadata, and saved figure recipes are cleared when the order changes so that the next export reflects the new order.

## Extending the Bile Acid Panel

Bile acid definitions live in [`config/bile_acid_species.py`](config/bile_acid_species.py). Add a new species there to make it available to structure detection, totals, percentages, ratios, and figures:

```python
BILE_ACID_PANEL["NEW_BA"] = BileAcidSpecies(
    abbreviation="NEW_BA",
    full_name="New Bile Acid",
    conjugation=Conjugation.UNCONJUGATED,
    origin=Origin.SECONDARY,
    core_structure=CoreStructure.OTHER,
    notes="Custom bile acid"
)
```

## Statistical Tests

The pipeline automatically selects tests based on data characteristics:

| Condition | 2 Groups | >2 Groups |
|-----------|----------|-----------|
| Normal + equal variance | Independent t-test | One-way ANOVA |
| Normal + unequal variance | Welch's t-test | Welch's ANOVA |
| Non-normal | Mann-Whitney U | Kruskal-Wallis |

Post-hoc tests (for >2 groups, only when the omnibus result is significant):

- Tukey HSD (parametric, equal variance)
- Games-Howell (parametric, unequal variance)
- Dunn's test (non-parametric)

Two-way and three-way designs use factorial ANOVA with interaction terms and factor-level summaries. When parametric assumptions fail, the pipeline uses the Aligned Rank Transform (ART). Two-way ART main effects and interactions are read from the full factorial model on aligned ranks, which gives the correct residual degrees of freedom.

### Multiple-testing control

Benjamini-Hochberg FDR correction is enabled by default for primary (omnibus) tests:

- Families are separate for each analysis section, raw vs log10 scale, and factorial effect.
- q-values control which omnibus results are declared significant.
- Post-hoc pairwise testing only runs for q-significant results.
- Pairwise post-hoc comparisons remain BH-adjusted within each endpoint (analyte, total, ratio, percentage, or hydrophobicity metric).

The FDR setting can be turned off in the sidebar to use unadjusted p-values.

### LOD handling and aggregate burden

Individual analytes are excluded from statistical testing when the below-LOD fraction reaches the **LOD exclusion threshold** (sidebar, default 50%), unless at least one group has enough detected values to satisfy the existing group-rescue rule.

Aggregate totals use a separate **aggregate LOD burden threshold** (sidebar, default 5%). The burden for an aggregate is

```text
R / (O + R)
```

where `O` is the sum of observed (non-below-LOD) contributions and `R` is the sum of the replacement contributions assigned to below-LOD cells at `LOD × dilution factor`. Exclusion is based on the **maximum group burden**, not on how many constituent bile acids were excluded individually.

- `total_all` is never excluded by this rule. A burden at or above the threshold produces a warning and is recorded in the report.
- Other aggregates are excluded when their maximum group burden reaches the threshold.
- Setting the aggregate threshold to `0` disables aggregate exclusion.
- Per-sample burdens, group summaries, and the overall burden are written to the `Aggregate_LOD_Burden` worksheet.

## Calculated Metrics

### Group Totals

- `total_all`: sum of all bile acids
- `total_primary`: sum of primary BAs
- `total_secondary`: sum of secondary BAs
- `total_conjugated`: sum of all conjugated BAs
- `total_unconjugated`: sum of unconjugated BAs
- `glycine_conjugated`: sum of glycine-conjugated BAs
- `taurine_conjugated`: sum of taurine-conjugated BAs

### Clinical Ratios

- `primary_to_secondary`
- `glycine_to_taurine`
- `conjugated_to_unconjugated`
- `TCA_to_GCA`, `GCA_to_TCA`
- `GCDCA_to_TCDCA`, `TCDCA_to_GCDCA`
- And more...

## What's New in 4.7.0

- Cross-analyte Benjamini-Hochberg FDR is on by default; q-values now determine omnibus significance and gate post-hoc testing
- Aggregate totals are screened using their LOD-bound replacement burden instead of the proportion of constituent BAs excluded
- `total_all` is always tested and can only receive a warning; other aggregates are excluded at a separate 5% burden threshold
- Per-sample and per-group aggregate burdens are written to the `Aggregate_LOD_Burden` report worksheet
- Category sheets reuse their corresponding totals statistics, so p- and q-values are identical without duplicate testing
- Two-way ART main effects now use the full factorial model on aligned ranks, fixing the residual degrees of freedom and F statistics
- Excel reports, in-app summaries, APA text, and figures display q-values and LOD warnings consistently
- Statistics remain available for raw and log10 scales, with separate FDR families for each

The lazy-section rendering, bounded-memory exports, and figure-recipe improvements from 4.6.2 remain in place.

## Citation

If you use this software, please cite the archived release:

> Way, G. W. (2026). *Bile Acid Analysis Pipeline* (Version 4.7.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21513592

The concept DOI above always resolves to the latest archived version. Version 4.6.1 is archived at https://doi.org/10.5281/zenodo.23483242. Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Project Structure

```text
bile_acid_pipeline/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── CITATION.cff                # Citation metadata
├── .zenodo.json                # Zenodo release metadata
├── config/
│   ├── __init__.py
│   └── bile_acid_species.py    # Bile acid definitions
├── modules/
│   ├── __init__.py
│   ├── data_processing.py      # Data loading and calculations
│   ├── report_generation.py    # Statistical report output
│   ├── statistical_tests.py    # Automated statistical analysis
│   └── visualization.py        # Publication-quality figures
└── tests/
    └── ...
```

## License

MIT License — free to use and modify for research purposes.
