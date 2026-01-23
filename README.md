# Bile Acid Analysis Pipeline

A comprehensive, user-friendly pipeline for analyzing LC-MS bile acid data with automated statistical testing and publication-quality visualizations.

## Features

- **Automatic data structure detection** - Recognizes 44 bile acid species from your LC-MS output
- **Smart statistical analysis** - Automatically tests assumptions (normality, homoscedasticity) and selects appropriate tests
- **Publication-quality figures** - Box plots, stacked bars, heatmaps, and more
- **Comprehensive calculations** - Group totals, percentages, clinical ratios
- **Export everything** - CSV files, figures in PNG/PDF, complete ZIP packages

## Installation

```bash
# Clone or download the pipeline
cd bile_acid_pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application (Streamlit)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Programmatic Usage

```python
from config import BILE_ACID_PANEL, get_primary, get_secondary
from modules import BileAcidDataProcessor, StatisticalAnalyzer, BileAcidVisualizer

# Process data
processor = BileAcidDataProcessor(lod_handling="zero")
processed = processor.load_and_process("your_data.xlsx")

# Run statistics
analyzer = StatisticalAnalyzer(alpha=0.05)
result = analyzer.analyze(processed.sample_data, "TCA", "Group")

# Create visualization
viz = BileAcidVisualizer()
fig, ax = viz.plot_group_comparison(processed.sample_data, "TCA", "Group")
fig.savefig("tca_comparison.png", dpi=300)
```

## Data Format

Expected input format:
- **Rows**: Samples
- **Columns**: Bile acid species (matching panel names)
- **First column(s)**: Sample ID, Group/Type
- **Values**: Concentrations (typically nmol/L)
- **Below LOD**: Can be "-----", "LOD", "BLQ", "ND", etc.

Example:
| Type | Sample_ID | TCA | GCA | TCDCA | GCDCA | ... |
|------|-----------|-----|-----|-------|-------|-----|
| HD-1 | 81-0210 | 37.62 | 194.1 | 231.9 | 3220.44 | ... |
| HD-2 | 81-0211 | 66.66 | 287.58 | 158.46 | 1534.02 | ... |
| AC-1 | 60-677 | 45.2 | ----- | 189.3 | 2890.1 | ... |

## Recognized Bile Acid Species

### Primary Bile Acids
- CA (Cholic Acid)
- CDCA (Chenodeoxycholic Acid)
- Conjugated forms: TCA, GCA, TCDCA, GCDCA

### Secondary Bile Acids
- DCA (Deoxycholic Acid)
- LCA (Lithocholic Acid)
- UDCA (Ursodeoxycholic Acid)
- And many more...

### Sulfated Bile Acids
- CA-3-S, CA-7-S, CDCA-3-S, DCA-3-S, LCA-3-S, UDCA-3-S

See `config/bile_acid_species.py` for the complete panel definition.

## Adding New Bile Acids

Edit `config/bile_acid_species.py`:

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
| Normal + Equal variance | Independent t-test | One-way ANOVA |
| Normal + Unequal variance | Welch's t-test | Welch's ANOVA |
| Non-normal | Mann-Whitney U | Kruskal-Wallis |

Post-hoc tests (for >2 groups):
- Tukey HSD (parametric, equal variance)
- Games-Howell (parametric, unequal variance)
- Dunn's test (non-parametric)

## Calculated Metrics

### Group Totals
- `total_all`: Sum of all bile acids
- `total_primary`: Sum of primary BAs
- `total_secondary`: Sum of secondary BAs
- `total_conjugated`: Sum of all conjugated BAs
- `total_unconjugated`: Sum of unconjugated BAs
- `glycine_conjugated`: Sum of glycine-conjugated BAs
- `taurine_conjugated`: Sum of taurine-conjugated BAs

### Clinical Ratios
- `primary_to_secondary`
- `glycine_to_taurine`
- `conjugated_to_unconjugated`
- `TCA_to_GCA`, `GCA_to_TCA`
- `GCDCA_to_TCDCA`, `TCDCA_to_GCDCA`
- And more...

## Project Structure

```
bile_acid_pipeline/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── config/
│   ├── __init__.py
│   └── bile_acid_species.py  # BA definitions (EDIT THIS FOR NEW BAs)
└── modules/
    ├── __init__.py
    ├── data_processing.py    # Data loading and calculations
    ├── report_generation.py  # Statistical report output
    ├── statistical_tests.py  # Automated statistical analysis
    └── visualization.py      # Publication-quality figures
```

## License

MIT License - Feel free to use and modify for research purposes.

