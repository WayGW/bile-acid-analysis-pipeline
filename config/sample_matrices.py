"""
Sample Matrix Configuration
============================

Defines limit of detection (LOD) values for different sample types/matrices.
Each matrix can have default LOD and per-bile-acid specific LODs.

To add a new sample matrix:
1. Add a new entry to SAMPLE_MATRICES dictionary
2. Specify the default_lod (in nmol/L)
3. Optionally specify per-bile-acid LODs in the 'specific_lods' dict

Example:
    'new_matrix': SampleMatrix(
        name='New Matrix',
        description='Description of the sample type',
        default_lod=0.5,
        dilution_factor=1,
        specific_lods={
            'TCA': 0.3,
            'GCA': 0.4,
        }
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class SampleMatrix:
    """Configuration for a sample matrix type."""
    name: str
    description: str
    default_lod: float  # Default LOD in nmol/L
    dilution_factor: float = 1.0  # Dilution factor applied to raw values
    units: str = "nmol/L"
    specific_lods: Dict[str, float] = field(default_factory=dict)  # Per-BA LODs
    
    def get_lod(self, bile_acid: str) -> float:
        """Get LOD for a specific bile acid, or default if not specified."""
        return self.specific_lods.get(bile_acid, self.default_lod)


# =============================================================================
# SAMPLE MATRIX DEFINITIONS
# =============================================================================
# Add new sample matrices here

SAMPLE_MATRICES: Dict[str, SampleMatrix] = {
    
    'serum': SampleMatrix(
        name='Serum',
        description='Human or animal serum samples',
        default_lod=0.3,  # nmol/L
        dilution_factor=6.0,  # Standard serum dilution
        units='nmol/L',
        specific_lods={
            # Add bile-acid specific LODs if they differ from default
            # 'TCA': 0.25,
            # 'GCA': 0.35,
        }
    ),
    
    'plasma': SampleMatrix(
        name='Plasma',
        description='Human or animal plasma samples',
        default_lod=0.3,
        dilution_factor=6.0,
        units='nmol/L',
        specific_lods={}
    ),
    
    'liver': SampleMatrix(
        name='Liver Tissue',
        description='Liver tissue homogenate',
        default_lod=1.0,
        dilution_factor=10.0,
        units='nmol/g tissue',
        specific_lods={}
    ),
    
    'bile': SampleMatrix(
        name='Bile',
        description='Gallbladder or hepatic bile',
        default_lod=10.0,  # Higher concentrations in bile
        dilution_factor=100.0,
        units='µmol/L',
        specific_lods={}
    ),
    
    'feces': SampleMatrix(
        name='Feces',
        description='Fecal samples',
        default_lod=5.0,
        dilution_factor=20.0,
        units='nmol/g',
        specific_lods={}
    ),
    
    'urine': SampleMatrix(
        name='Urine',
        description='Urine samples',
        default_lod=0.5,
        dilution_factor=2.0,
        units='nmol/L',
        specific_lods={}
    ),
    
    'cell_culture': SampleMatrix(
        name='Cell Culture Media',
        description='In vitro cell culture supernatant',
        default_lod=0.1,
        dilution_factor=1.0,
        units='nmol/L',
        specific_lods={}
    ),
    
    'custom': SampleMatrix(
        name='Custom',
        description='User-defined matrix with custom LOD',
        default_lod=0.3,
        dilution_factor=1.0,
        units='nmol/L',
        specific_lods={}
    ),
}


def get_matrix(matrix_name: str) -> SampleMatrix:
    """Get a sample matrix configuration by name."""
    return SAMPLE_MATRICES.get(matrix_name.lower(), SAMPLE_MATRICES['custom'])


def get_available_matrices() -> List[str]:
    """Get list of available matrix names."""
    return list(SAMPLE_MATRICES.keys())


def get_matrix_display_names() -> Dict[str, str]:
    """Get mapping of matrix keys to display names."""
    return {key: matrix.name for key, matrix in SAMPLE_MATRICES.items()}


def add_custom_matrix(
    key: str,
    name: str,
    description: str,
    default_lod: float,
    dilution_factor: float = 1.0,
    units: str = "nmol/L",
    specific_lods: Optional[Dict[str, float]] = None
) -> SampleMatrix:
    """
    Add a new custom sample matrix at runtime.
    
    Args:
        key: Unique identifier for the matrix
        name: Display name
        description: Description of the sample type
        default_lod: Default limit of detection
        dilution_factor: Dilution factor for raw values
        units: Concentration units
        specific_lods: Per-bile-acid LOD overrides
        
    Returns:
        The created SampleMatrix object
    """
    matrix = SampleMatrix(
        name=name,
        description=description,
        default_lod=default_lod,
        dilution_factor=dilution_factor,
        units=units,
        specific_lods=specific_lods or {}
    )
    SAMPLE_MATRICES[key.lower()] = matrix
    return matrix


if __name__ == "__main__":
    # Demo
    print("Available Sample Matrices:")
    print("=" * 50)
    for key, matrix in SAMPLE_MATRICES.items():
        print(f"\n{matrix.name} ({key}):")
        print(f"  Description: {matrix.description}")
        print(f"  Default LOD: {matrix.default_lod} {matrix.units}")
        print(f"  Dilution Factor: {matrix.dilution_factor}x")
        if matrix.specific_lods:
            print(f"  Specific LODs: {matrix.specific_lods}")
