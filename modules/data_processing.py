"""
Data Processing Module for Bile Acid Analysis
===============================================

Handles:
1. Loading data from Excel/ODS files
2. Detecting and parsing data structure
3. Data cleaning (handling LOD, missing values)
4. Normalization and calculations
5. Aggregation by bile acid groups
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import re
import warnings

# Import bile acid configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.bile_acid_species import (
    BILE_ACID_PANEL, ANALYSIS_GROUPS, CLINICAL_RATIOS,
    get_all_species, validate_columns, get_species_info,
    Conjugation, Origin
)


@dataclass
class DataStructureInfo:
    """Information about detected data structure."""
    n_rows: int
    n_cols: int
    sample_id_col: Optional[str] = None
    group_col: Optional[str] = None
    other_metadata_cols: List[str] = field(default_factory=list)
    bile_acid_cols: List[str] = field(default_factory=list)
    unrecognized_cols: List[str] = field(default_factory=list)
    has_standards: bool = False
    standard_rows: List[int] = field(default_factory=list)
    data_start_row: int = 0
    units: str = "nmol/L"
    sheet_used: Optional[str] = None  # Which sheet the data was loaded from


@dataclass
class ProcessedData:
    """Container for processed bile acid data."""
    # Core data
    raw_data: pd.DataFrame
    sample_data: pd.DataFrame  # Samples only, no standards
    
    # Structure info
    structure: DataStructureInfo
    
    # Calculated values
    concentrations: pd.DataFrame  # Individual BA concentrations
    percentages: pd.DataFrame     # % of total for each BA
    totals: pd.DataFrame          # Group totals (conjugated, primary, etc.)
    ratios: pd.DataFrame          # Clinical ratios
    
    # Summary statistics by group
    group_summaries: Optional[Dict[str, pd.DataFrame]] = None


class BileAcidDataProcessor:
    """
    Process bile acid LC-MS data from Excel files.
    
    Usage:
        processor = BileAcidDataProcessor()
        processed = processor.load_and_process("data.xlsx")
    """
    
    # Common patterns for identifying columns
    SAMPLE_ID_PATTERNS = [
        r'^sample[\s_-]*(id|name|#)?$',
        r'^name$',
        r'^id$',
        r'^specimen$',
    ]
    
    GROUP_PATTERNS = [
        r'^(group|type|category|class|condition|treatment)$',
        r'^sample[\s_-]*type$',
    ]
    
    STANDARD_PATTERNS = [
        r'std[\s_-]*\d+',
        r'standard',
        r'cal[\s_-]*\d+',
        r'calibr',
        r'qc[\s_-]*\d*',
        r'blank',
    ]
    
    LOD_VALUES = ['-----', 'LOD', 'BLQ', 'ND', 'N/D', '<LOD', '<LOQ', 'BLOQ']
    
    def __init__(
        self,
        lod_handling: str = "zero",  # "zero", "half_min", "drop"
        lod_value: float = 0.1,  # Default LOD value for ratio calculations
        custom_bile_acids: Optional[Dict] = None
    ):
        """
        Initialize the processor.
        
        Args:
            lod_handling: How to handle below-LOD values
                - "zero": Replace with 0
                - "half_min": Replace with half the minimum detected value
                - "drop": Keep as NaN
            lod_value: Limit of detection value (used for ratio calculations)
            custom_bile_acids: Additional bile acid definitions to merge with panel
        """
        self.lod_handling = lod_handling
        self.lod_value = lod_value
        
        # Merge custom BAs if provided
        self.bile_acid_panel = BILE_ACID_PANEL.copy()
        if custom_bile_acids:
            self.bile_acid_panel.update(custom_bile_acids)
    
    # Preferred sheet names for processed data (in order of preference)
    PREFERRED_SHEET_PATTERNS = [
        r'serum[\s_-]*c',
        r'plasma[\s_-]*c', 
        r'processed',
        r'corrected',
        r'final',
        r'results',
    ]
    
    def load_file(
        self, 
        filepath: Union[str, Path],
        sheet_name: Union[str, int, None] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Load data from Excel/ODS file.
        
        Args:
            filepath: Path to the file
            sheet_name: Specific sheet to load. If None, auto-detects best sheet.
            
        Returns:
            Tuple of (DataFrame, sheet_name_used)
        """
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.ods':
            engine = 'odf'
        elif filepath.suffix.lower() in ['.xlsx', '.xlsm']:
            engine = 'openpyxl'
        elif filepath.suffix.lower() == '.xls':
            engine = 'xlrd'
        else:
            engine = None
        
        # Get list of sheet names
        xlsx = pd.ExcelFile(filepath, engine=engine)
        available_sheets = xlsx.sheet_names
        
        # Determine which sheet to use
        if sheet_name is not None:
            # User specified a sheet
            selected_sheet = sheet_name
        else:
            # Auto-detect: look for preferred sheet names
            selected_sheet = self._find_best_sheet(available_sheets)
        
        df = pd.read_excel(filepath, sheet_name=selected_sheet, header=None, engine=engine)
        
        # Return both the dataframe and which sheet was used
        sheet_used = selected_sheet if isinstance(selected_sheet, str) else available_sheets[selected_sheet]
        return df, sheet_used
    
    def _find_best_sheet(self, sheet_names: List[str]) -> Union[str, int]:
        """
        Find the best sheet to use based on naming patterns.
        
        Prefers sheets like 'Serum C' over raw 'LC-MS data'.
        Avoids sheets that look like analysis output (Overview, Report, etc.)
        """
        # Sheets to avoid (analysis output sheets)
        avoid_patterns = [
            r'overview',
            r'report',
            r'summary',
            r'total[\s_-]*(all|primary|secondary|conjugated)',
            r'glycine',
            r'taurine',
            r'sulfated',
        ]
        
        # Filter out sheets to avoid
        valid_sheets = []
        for sheet in sheet_names:
            should_avoid = any(re.search(p, sheet, re.IGNORECASE) for p in avoid_patterns)
            if not should_avoid:
                valid_sheets.append(sheet)
        
        # If no valid sheets remain, use all sheets
        if not valid_sheets:
            valid_sheets = sheet_names
        
        # Check for preferred patterns among valid sheets
        for pattern in self.PREFERRED_SHEET_PATTERNS:
            for sheet in valid_sheets:
                if re.search(pattern, sheet, re.IGNORECASE):
                    return sheet
        
        # If only one valid sheet, use it
        if len(valid_sheets) == 1:
            return valid_sheets[0]
        
        # If multiple sheets and no match, prefer second sheet 
        # (often first is raw LC-MS, second is processed)
        if len(valid_sheets) >= 2:
            # But check if first sheet looks like raw data
            first_lower = valid_sheets[0].lower()
            if 'lc-ms' in first_lower or 'raw' in first_lower or 'data' in first_lower:
                return valid_sheets[1]
        
        # Default to first valid sheet
        return valid_sheets[0] if valid_sheets else 0
    
    def get_available_sheets(self, filepath: Union[str, Path]) -> List[str]:
        """Get list of available sheet names in a file."""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.ods':
            engine = 'odf'
        elif filepath.suffix.lower() in ['.xlsx', '.xlsm']:
            engine = 'openpyxl'
        elif filepath.suffix.lower() == '.xls':
            engine = 'xlrd'
        else:
            engine = None
        
        xlsx = pd.ExcelFile(filepath, engine=engine)
        return xlsx.sheet_names
    
    def detect_structure(self, df: pd.DataFrame) -> DataStructureInfo:
        """
        Auto-detect the structure of the data file.
        
        Identifies:
        - Header row
        - Sample ID and group columns
        - Bile acid columns
        - Standard/calibration rows
        """
        info = DataStructureInfo(n_rows=len(df), n_cols=len(df.columns))
        
        # Find header row (row containing bile acid names)
        header_row = self._find_header_row(df)
        info.data_start_row = header_row + 1
        
        # Get column names
        if header_row >= 0:
            headers = df.iloc[header_row].astype(str).tolist()
            # Handle multi-row headers
            if header_row > 0:
                # Check if row above has units info
                prev_row = df.iloc[header_row - 1].astype(str).tolist()
                for i, (h, p) in enumerate(zip(headers, prev_row)):
                    if 'nmol' in str(p).lower() or 'conc' in str(p).lower():
                        info.units = str(p)
                        break
        else:
            headers = [f"col_{i}" for i in range(len(df.columns))]
        
        # Classify columns
        known_bas = set(self.bile_acid_panel.keys())
        
        for i, col in enumerate(headers):
            col_str = str(col).strip()
            col_lower = col_str.lower()
            
            # Check for sample ID
            if any(re.match(p, col_lower, re.I) for p in self.SAMPLE_ID_PATTERNS):
                info.sample_id_col = col_str
                continue
            
            # Check for group column
            if any(re.match(p, col_lower, re.I) for p in self.GROUP_PATTERNS):
                info.group_col = col_str
                continue
            
            # Check for bile acid
            if col_str in known_bas:
                info.bile_acid_cols.append(col_str)
            elif col_str not in ['nan', '', 'NaN', 'None']:
                # Check for partial matches (handle slight naming variations)
                matched = False
                for ba in known_bas:
                    if col_str.replace('_', '').replace('-', '').lower() == \
                       ba.replace('_', '').replace('-', '').lower():
                        info.bile_acid_cols.append(ba)
                        matched = True
                        break
                
                if not matched and not pd.isna(col):
                    info.unrecognized_cols.append(col_str)
        
        # If no group col found, check first column for group patterns
        if info.group_col is None and len(df.columns) > 0:
            first_col_vals = df.iloc[info.data_start_row:, 0].astype(str)
            if self._looks_like_group_column(first_col_vals):
                info.group_col = headers[0] if headers[0] not in ['nan', ''] else 'Type'
        
        # Find standard/calibration rows
        for idx in range(info.data_start_row, len(df)):
            row_val = str(df.iloc[idx, 1] if len(df.columns) > 1 else df.iloc[idx, 0]).lower()
            if any(re.search(p, row_val, re.I) for p in self.STANDARD_PATTERNS):
                info.standard_rows.append(idx)
                info.has_standards = True
        
        return info
    
    def _find_header_row(self, df: pd.DataFrame) -> int:
        """Find the row containing column headers (bile acid names)."""
        known_bas = set(self.bile_acid_panel.keys())
        
        for idx in range(min(10, len(df))):  # Check first 10 rows
            row_vals = set(str(v).strip() for v in df.iloc[idx].values if pd.notna(v))
            matches = row_vals & known_bas
            if len(matches) >= 3:  # At least 3 bile acid names
                return idx
        
        return 0  # Default to first row
    
    def _looks_like_group_column(self, values: pd.Series) -> bool:
        """Check if a column looks like group labels."""
        # Group columns typically have repeated values and are short strings
        unique_vals = values.dropna().unique()
        
        if len(unique_vals) < 2:
            return False
        if len(unique_vals) > len(values) * 0.5:
            return False  # Too many unique values
        
        # Check if values look like group names (short alphanumeric)
        for v in unique_vals[:10]:
            if len(str(v)) > 50:
                return False
            if re.match(r'^[\d.]+$', str(v)):
                return False  # Pure numbers aren't group names
        
        return True
    
    def clean_data(
        self, 
        df: pd.DataFrame, 
        structure: DataStructureInfo
    ) -> pd.DataFrame:
        """
        Clean the raw data:
        - Set proper column names
        - Remove standard rows
        - Handle LOD values
        - Convert to numeric
        - Remove rows with NaN/empty group values
        """
        # Get header row and set columns
        header_row = structure.data_start_row - 1
        if header_row >= 0:
            df.columns = df.iloc[header_row].astype(str).str.strip()
            df = df.iloc[structure.data_start_row:].reset_index(drop=True)
        
        # Remove standard rows (adjust indices after removing header)
        adjusted_std_rows = [r - structure.data_start_row for r in structure.standard_rows]
        adjusted_std_rows = [r for r in adjusted_std_rows if r >= 0]
        if adjusted_std_rows:
            df = df.drop(index=adjusted_std_rows).reset_index(drop=True)
        
        # Handle LOD values and convert to numeric
        for col in structure.bile_acid_cols:
            if col in df.columns:
                df[col] = self._clean_numeric_column(df[col], structure)
        
        # Remove rows where group column is NaN or empty
        if structure.group_col and structure.group_col in df.columns:
            # Convert to string and check for empty/nan values
            group_series = df[structure.group_col].astype(str).str.strip()
            valid_mask = (
                df[structure.group_col].notna() & 
                (group_series != '') & 
                (group_series.str.lower() != 'nan') &
                (group_series.str.lower() != 'none')
            )
            df = df[valid_mask].reset_index(drop=True)
        
        return df
    
    def _clean_numeric_column(
        self, 
        series: pd.Series,
        structure: DataStructureInfo
    ) -> pd.Series:
        """
        Clean a single numeric column.
        
        Handles below-LOD markers (like '-----', 'LOD', 'BLQ') based on lod_handling:
        - "zero": Replace with 0 (legacy behavior)
        - "lod": Replace with the LOD value from sample matrix
        - "half_lod": Replace with half the LOD value
        - "half_min": Replace with half the minimum detected value
        - "drop": Keep as NaN (will exclude from calculations)
        """
        # Convert series to object type first to avoid downcasting issues
        series = series.astype(object)
        
        # Replace LOD marker values with None (will become NaN after to_numeric)
        lod_values_all = self.LOD_VALUES + [v.lower() for v in self.LOD_VALUES]
        mask = series.isin(lod_values_all)
        series = series.where(~mask, None)
        
        # Convert to numeric
        series = pd.to_numeric(series, errors='coerce')
        
        # Handle below-LOD values based on strategy
        if self.lod_handling == "zero":
            series = series.fillna(0)
        elif self.lod_handling == "lod":
            # Use the LOD value from sample matrix
            series = series.fillna(self.lod_value)
        elif self.lod_handling == "half_lod":
            # Use half the LOD value
            series = series.fillna(self.lod_value / 2)
        elif self.lod_handling == "half_min":
            min_val = series[series > 0].min()
            if pd.notna(min_val):
                series = series.fillna(min_val / 2)
            else:
                series = series.fillna(self.lod_value / 2)
        # "drop" leaves NaN
        
        return series
    
    def calculate_totals(
        self, 
        df: pd.DataFrame,
        bile_acid_cols: List[str]
    ) -> pd.DataFrame:
        """Calculate aggregate totals for bile acid groups."""
        totals = pd.DataFrame(index=df.index)
        
        # Get available columns
        available_cols = set(df.columns) & set(bile_acid_cols)
        
        for group_name, species_list in ANALYSIS_GROUPS.items():
            cols_in_group = [c for c in species_list if c in available_cols]
            if cols_in_group:
                totals[group_name] = df[cols_in_group].sum(axis=1)
        
        # Total of all bile acids
        totals['total_all'] = df[list(available_cols)].sum(axis=1)
        
        return totals
    
    def calculate_percentages(
        self, 
        df: pd.DataFrame,
        bile_acid_cols: List[str]
    ) -> pd.DataFrame:
        """Calculate percentage of total for each bile acid."""
        available_cols = [c for c in bile_acid_cols if c in df.columns]
        
        total = df[available_cols].sum(axis=1)
        
        pct_df = pd.DataFrame(index=df.index)
        for col in available_cols:
            pct_df[f"{col}_pct"] = (df[col] / total * 100).replace([np.inf, -np.inf], 0)
        
        return pct_df
    
    def calculate_ratios(
        self, 
        df: pd.DataFrame,
        bile_acid_cols: List[str],
        lod_value: float = 0.1
    ) -> pd.DataFrame:
        """
        Calculate clinical ratios.
        
        Note: Below-LOD values should already be replaced with LOD value
        during data cleaning. This function handles any remaining zeros
        as a safety measure.
        
        Args:
            df: DataFrame with bile acid concentrations
            bile_acid_cols: List of bile acid column names
            lod_value: Fallback LOD value if denominator is still zero
        """
        ratios = pd.DataFrame(index=df.index)
        available_cols = set(df.columns) & set(bile_acid_cols)
        
        for ratio_name, ratio_def in CLINICAL_RATIOS.items():
            num_cols = [c for c in ratio_def['numerator'] if c in available_cols]
            den_cols = [c for c in ratio_def['denominator'] if c in available_cols]
            
            if num_cols and den_cols:
                numerator = df[num_cols].sum(axis=1)
                denominator = df[den_cols].sum(axis=1)
                
                # Safety: replace any remaining zeros with LOD value
                denominator_safe = denominator.replace(0, lod_value)
                
                # Calculate ratio
                ratios[ratio_name] = numerator / denominator_safe
        
        return ratios
    
    def calculate_group_summaries(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_cols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Calculate summary statistics for each group."""
        if group_col not in df.columns:
            return {}
        
        summaries = {}
        
        for col in value_cols:
            if col not in df.columns:
                continue
            
            summary = df.groupby(group_col)[col].agg([
                ('n', 'count'),
                ('mean', 'mean'),
                ('std', 'std'),
                ('sem', 'sem'),
                ('median', 'median'),
                ('min', 'min'),
                ('max', 'max')
            ]).round(4)
            
            # Add percentage of group total
            group_totals = df.groupby(group_col)[value_cols].sum()
            if col in group_totals.columns:
                summary['pct_of_total'] = (
                    df.groupby(group_col)[col].sum() / 
                    group_totals.sum(axis=1) * 100
                ).round(2)
            
            summaries[col] = summary
        
        return summaries
    
    def load_and_process(
        self,
        filepath: Union[str, Path],
        sheet_name: Union[str, int, None] = None,
        group_col: Optional[str] = None
    ) -> ProcessedData:
        """
        Main entry point: load and fully process a bile acid data file.
        
        Args:
            filepath: Path to Excel/ODS file
            sheet_name: Sheet to load (name or index). If None, auto-detects best sheet.
            group_col: Override auto-detected group column
            
        Returns:
            ProcessedData object with all calculations
        """
        # Load raw data (auto-detects best sheet if not specified)
        raw_df, sheet_used = self.load_file(filepath, sheet_name)
        
        # Detect structure
        structure = self.detect_structure(raw_df)
        
        # Store which sheet was used
        structure.sheet_used = sheet_used
        
        # Override group col if specified
        if group_col:
            structure.group_col = group_col
        
        # Clean data
        clean_df = self.clean_data(raw_df.copy(), structure)
        
        # Extract sample data (with metadata)
        sample_df = clean_df.copy()
        
        # Get concentration data (only BA columns)
        ba_cols = [c for c in structure.bile_acid_cols if c in sample_df.columns]
        concentrations = sample_df[ba_cols].copy()
        
        # Calculate derived values
        percentages = self.calculate_percentages(sample_df, ba_cols)
        totals = self.calculate_totals(sample_df, ba_cols)
        ratios = self.calculate_ratios(sample_df, ba_cols, lod_value=self.lod_value)
        
        # Group summaries if group column exists
        group_summaries = None
        if structure.group_col and structure.group_col in sample_df.columns:
            all_value_cols = ba_cols + list(totals.columns) + list(ratios.columns)
            
            # Merge for summary calculation
            full_df = pd.concat([sample_df, totals, ratios], axis=1)
            group_summaries = self.calculate_group_summaries(
                full_df, 
                structure.group_col,
                all_value_cols
            )
        
        return ProcessedData(
            raw_data=raw_df,
            sample_data=sample_df,
            structure=structure,
            concentrations=concentrations,
            percentages=percentages,
            totals=totals,
            ratios=ratios,
            group_summaries=group_summaries
        )
    
    def get_analysis_dataframe(
        self,
        processed: ProcessedData,
        include_percentages: bool = True,
        include_totals: bool = True,
        include_ratios: bool = True
    ) -> pd.DataFrame:
        """
        Combine all processed data into a single analysis-ready DataFrame.
        """
        dfs = [processed.sample_data]
        
        if include_percentages:
            dfs.append(processed.percentages)
        if include_totals:
            dfs.append(processed.totals)
        if include_ratios:
            dfs.append(processed.ratios)
        
        return pd.concat(dfs, axis=1)


def validate_data_quality(processed: ProcessedData) -> Dict[str, Any]:
    """
    Generate a data quality report.
    """
    report = {
        'n_samples': len(processed.sample_data),
        'n_bile_acids_detected': len(processed.structure.bile_acid_cols),
        'n_unrecognized_cols': len(processed.structure.unrecognized_cols),
        'unrecognized_cols': processed.structure.unrecognized_cols,
        'group_col_detected': processed.structure.group_col,
        'groups': None,
        'missing_data': {},
        'zero_prevalence': {},
    }
    
    # Group info
    if processed.structure.group_col:
        groups = processed.sample_data[processed.structure.group_col].unique()
        report['groups'] = list(groups)
        report['n_groups'] = len(groups)
    
    # Missing data per BA
    for col in processed.structure.bile_acid_cols:
        if col in processed.concentrations.columns:
            n_missing = processed.concentrations[col].isna().sum()
            n_zero = (processed.concentrations[col] == 0).sum()
            report['missing_data'][col] = n_missing
            report['zero_prevalence'][col] = n_zero / len(processed.concentrations) * 100
    
    return report


if __name__ == "__main__":
    # Demo with sample file
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print("Usage: python data_processing.py <path_to_file>")
        print("\nRunning demo with synthetic data...")
        
        # Create synthetic demo data
        np.random.seed(42)
        demo_data = pd.DataFrame({
            'Type': ['HD']*5 + ['AC']*5,
            'Sample_ID': [f'S{i}' for i in range(10)],
            'TCA': np.random.lognormal(4, 1, 10),
            'GCA': np.random.lognormal(5, 1, 10),
            'TCDCA': np.random.lognormal(4.5, 1, 10),
            'GCDCA': np.random.lognormal(5.5, 1, 10),
            'CA': np.random.lognormal(3, 1, 10),
            'CDCA': np.random.lognormal(4, 1, 10),
            'DCA': np.random.lognormal(3.5, 1, 10),
            'LCA': np.random.lognormal(2, 1, 10),
        })
        
        # Save to temp file
        demo_path = '/tmp/demo_ba_data.xlsx'
        demo_data.to_excel(demo_path, index=False)
        filepath = demo_path
    
    # Process file
    processor = BileAcidDataProcessor()
    processed = processor.load_and_process(filepath)
    
    # Print report
    print("\n" + "="*60)
    print("DATA PROCESSING REPORT")
    print("="*60)
    
    quality = validate_data_quality(processed)
    print(f"\nSamples: {quality['n_samples']}")
    print(f"Bile acids detected: {quality['n_bile_acids_detected']}")
    print(f"Group column: {quality['group_col_detected']}")
    print(f"Groups: {quality['groups']}")
    
    if quality['unrecognized_cols']:
        print(f"\nUnrecognized columns: {quality['unrecognized_cols']}")
    
    print("\n--- TOTALS (first 5 rows) ---")
    print(processed.totals.head())
    
    print("\n--- RATIOS (first 5 rows) ---")
    print(processed.ratios.head())
