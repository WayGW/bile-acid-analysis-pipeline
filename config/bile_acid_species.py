"""
Bile Acid Species Configuration
===============================

This module defines all bile acid species and their classifications.
Update this file when new bile acids are added to the LC-MS panel.

Classification Schema:
- Conjugation: unconjugated, glycine-conjugated, taurine-conjugated, sulfated
- Origin: primary (synthesized in liver), secondary (bacterial modification)
- Core structure: CA (cholic acid), CDCA (chenodeoxycholic acid), DCA (deoxycholic acid), 
                  LCA (lithocholic acid), UDCA (ursodeoxycholic acid), MCA (muricholic acid)

Last updated: 2024-07-02
Panel source: IU BA Zhou LC-MS Panel
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum


class Conjugation(Enum):
    """Bile acid conjugation status."""
    UNCONJUGATED = "unconjugated"
    GLYCINE = "glycine"
    TAURINE = "taurine"
    SULFATED = "sulfated"
    N_ACYL = "n-acyl"  # NCA, NDCA


class Origin(Enum):
    """Bile acid biosynthetic origin."""
    PRIMARY = "primary"
    SECONDARY = "secondary"


class CoreStructure(Enum):
    """Core bile acid structure/family."""
    CA = "cholic_acid"
    CDCA = "chenodeoxycholic_acid"
    DCA = "deoxycholic_acid"
    LCA = "lithocholic_acid"
    UDCA = "ursodeoxycholic_acid"
    MCA = "muricholic_acid"  # murine, but can appear in humans
    HCA = "hyocholic_acid"
    HDCA = "hyodeoxycholic_acid"
    MDCA = "murideoxycholic_acid"
    KETO = "keto_derivative"
    OTHER = "other"


@dataclass
class BileAcidSpecies:
    """Definition of a single bile acid species."""
    abbreviation: str
    full_name: str
    conjugation: Conjugation
    origin: Origin
    core_structure: CoreStructure
    hydroxyl_positions: List[int] = field(default_factory=list)
    is_keto_derivative: bool = False
    is_iso_form: bool = False
    notes: str = ""


# =============================================================================
# BILE ACID PANEL DEFINITION
# =============================================================================
# To add a new bile acid, simply add a new entry to this dictionary.
# The key should match the column name in your LC-MS output.

BILE_ACID_PANEL: Dict[str, BileAcidSpecies] = {
    
    # -------------------------------------------------------------------------
    # UNCONJUGATED PRIMARY BILE ACIDS
    # -------------------------------------------------------------------------
    "CA": BileAcidSpecies(
        abbreviation="CA",
        full_name="Cholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CA,
        hydroxyl_positions=[3, 7, 12],
    ),
    "CDCA": BileAcidSpecies(
        abbreviation="CDCA",
        full_name="Chenodeoxycholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CDCA,
        hydroxyl_positions=[3, 7],
    ),
    
    # -------------------------------------------------------------------------
    # UNCONJUGATED SECONDARY BILE ACIDS
    # -------------------------------------------------------------------------
    "DCA": BileAcidSpecies(
        abbreviation="DCA",
        full_name="Deoxycholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.DCA,
        hydroxyl_positions=[3, 12],
        notes="Derived from CA by 7-dehydroxylation",
    ),
    "LCA": BileAcidSpecies(
        abbreviation="LCA",
        full_name="Lithocholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.LCA,
        hydroxyl_positions=[3],
        notes="Derived from CDCA by 7-dehydroxylation",
    ),
    "UDCA": BileAcidSpecies(
        abbreviation="UDCA",
        full_name="Ursodeoxycholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.UDCA,
        hydroxyl_positions=[3, 7],
        notes="7-beta epimer of CDCA",
    ),
    "isoDCA": BileAcidSpecies(
        abbreviation="isoDCA",
        full_name="Isodeoxycholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.DCA,
        hydroxyl_positions=[3, 12],
        is_iso_form=True,
    ),
    "isoLCA": BileAcidSpecies(
        abbreviation="isoLCA",
        full_name="Isolithocholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.LCA,
        hydroxyl_positions=[3],
        is_iso_form=True,
    ),
    "allo_isoLCA": BileAcidSpecies(
        abbreviation="allo_isoLCA",
        full_name="Allo-isolithocholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.LCA,
        hydroxyl_positions=[3],
        is_iso_form=True,
        notes="5-alpha configuration",
    ),
    "HCA": BileAcidSpecies(
        abbreviation="HCA",
        full_name="Hyocholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.HCA,
        hydroxyl_positions=[3, 6, 7],
    ),
    "HDCA": BileAcidSpecies(
        abbreviation="HDCA",
        full_name="Hyodeoxycholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.HDCA,
        hydroxyl_positions=[3, 6],
    ),
    "MDCA": BileAcidSpecies(
        abbreviation="MDCA",
        full_name="Murideoxycholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.MDCA,
        hydroxyl_positions=[3, 6],
    ),
    
    # -------------------------------------------------------------------------
    # UNCONJUGATED MURICHOLIC ACIDS (trace in humans)
    # -------------------------------------------------------------------------
    "wMCA": BileAcidSpecies(
        abbreviation="wMCA",
        full_name="Omega-Muricholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.MCA,
        hydroxyl_positions=[3, 6, 7],
    ),
    "aMCA": BileAcidSpecies(
        abbreviation="aMCA",
        full_name="Alpha-Muricholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.MCA,
        hydroxyl_positions=[3, 6, 7],
    ),
    "bMCA": BileAcidSpecies(
        abbreviation="bMCA",
        full_name="Beta-Muricholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.MCA,
        hydroxyl_positions=[3, 6, 7],
    ),
    
    # -------------------------------------------------------------------------
    # KETO DERIVATIVES (UNCONJUGATED)
    # -------------------------------------------------------------------------
    "7keto_DCA": BileAcidSpecies(
        abbreviation="7keto_DCA",
        full_name="7-Ketodeoxycholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.DCA,
        is_keto_derivative=True,
    ),
    "7keto_LCA": BileAcidSpecies(
        abbreviation="7keto_LCA",
        full_name="7-Ketolithocholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.LCA,
        is_keto_derivative=True,
    ),
    "12keto_LCA": BileAcidSpecies(
        abbreviation="12keto_LCA",
        full_name="12-Ketolithocholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.LCA,
        is_keto_derivative=True,
    ),
    "3KetoLCA": BileAcidSpecies(
        abbreviation="3KetoLCA",
        full_name="3-Ketolithocholic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.LCA,
        is_keto_derivative=True,
    ),
    "3Keto,7a,12a(OH)2": BileAcidSpecies(
        abbreviation="3Keto,7a,12a(OH)2",
        full_name="3-Keto-7α,12α-dihydroxycholanic Acid",
        conjugation=Conjugation.UNCONJUGATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.CA,
        is_keto_derivative=True,
        notes="Oxidized CA derivative",
    ),
    
    # -------------------------------------------------------------------------
    # GLYCINE-CONJUGATED BILE ACIDS
    # -------------------------------------------------------------------------
    "GCA": BileAcidSpecies(
        abbreviation="GCA",
        full_name="Glycocholic Acid",
        conjugation=Conjugation.GLYCINE,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CA,
        hydroxyl_positions=[3, 7, 12],
    ),
    "GCDCA": BileAcidSpecies(
        abbreviation="GCDCA",
        full_name="Glycochenodeoxycholic Acid",
        conjugation=Conjugation.GLYCINE,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CDCA,
        hydroxyl_positions=[3, 7],
    ),
    "GDCA": BileAcidSpecies(
        abbreviation="GDCA",
        full_name="Glycodeoxycholic Acid",
        conjugation=Conjugation.GLYCINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.DCA,
        hydroxyl_positions=[3, 12],
    ),
    "GLCA": BileAcidSpecies(
        abbreviation="GLCA",
        full_name="Glycolithocholic Acid",
        conjugation=Conjugation.GLYCINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.LCA,
        hydroxyl_positions=[3],
    ),
    "GUDCA": BileAcidSpecies(
        abbreviation="GUDCA",
        full_name="Glycoursodeoxycholic Acid",
        conjugation=Conjugation.GLYCINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.UDCA,
        hydroxyl_positions=[3, 7],
    ),
    "GHCA": BileAcidSpecies(
        abbreviation="GHCA",
        full_name="Glycohyocholic Acid",
        conjugation=Conjugation.GLYCINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.HCA,
        hydroxyl_positions=[3, 6, 7],
    ),
    "GHDCA": BileAcidSpecies(
        abbreviation="GHDCA",
        full_name="Glycohyodeoxycholic Acid",
        conjugation=Conjugation.GLYCINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.HDCA,
        hydroxyl_positions=[3, 6],
    ),
    "GbMCA": BileAcidSpecies(
        abbreviation="GbMCA",
        full_name="Glyco-beta-Muricholic Acid",
        conjugation=Conjugation.GLYCINE,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.MCA,
        hydroxyl_positions=[3, 6, 7],
    ),
    
    # -------------------------------------------------------------------------
    # TAURINE-CONJUGATED BILE ACIDS
    # -------------------------------------------------------------------------
    "TCA": BileAcidSpecies(
        abbreviation="TCA",
        full_name="Taurocholic Acid",
        conjugation=Conjugation.TAURINE,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CA,
        hydroxyl_positions=[3, 7, 12],
    ),
    "TCDCA": BileAcidSpecies(
        abbreviation="TCDCA",
        full_name="Taurochenodeoxycholic Acid",
        conjugation=Conjugation.TAURINE,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CDCA,
        hydroxyl_positions=[3, 7],
    ),
    "TDCA": BileAcidSpecies(
        abbreviation="TDCA",
        full_name="Taurodeoxycholic Acid",
        conjugation=Conjugation.TAURINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.DCA,
        hydroxyl_positions=[3, 12],
    ),
    "TLCA": BileAcidSpecies(
        abbreviation="TLCA",
        full_name="Taurolithocholic Acid",
        conjugation=Conjugation.TAURINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.LCA,
        hydroxyl_positions=[3],
    ),
    "TUDCA": BileAcidSpecies(
        abbreviation="TUDCA",
        full_name="Tauroursodeoxycholic Acid",
        conjugation=Conjugation.TAURINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.UDCA,
        hydroxyl_positions=[3, 7],
    ),
    "THDCA": BileAcidSpecies(
        abbreviation="THDCA",
        full_name="Taurohyodeoxycholic Acid",
        conjugation=Conjugation.TAURINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.HDCA,
        hydroxyl_positions=[3, 6],
    ),
    "TwMCA": BileAcidSpecies(
        abbreviation="TwMCA",
        full_name="Tauro-omega-Muricholic Acid",
        conjugation=Conjugation.TAURINE,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.MCA,
        hydroxyl_positions=[3, 6, 7],
    ),
    "TaMCA": BileAcidSpecies(
        abbreviation="TaMCA",
        full_name="Tauro-alpha-Muricholic Acid",
        conjugation=Conjugation.TAURINE,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.MCA,
        hydroxyl_positions=[3, 6, 7],
    ),
    "TbMCA": BileAcidSpecies(
        abbreviation="TbMCA",
        full_name="Tauro-beta-Muricholic Acid",
        conjugation=Conjugation.TAURINE,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.MCA,
        hydroxyl_positions=[3, 6, 7],
    ),
    
    # -------------------------------------------------------------------------
    # SULFATED BILE ACIDS
    # -------------------------------------------------------------------------
    "CA-7-S": BileAcidSpecies(
        abbreviation="CA-7-S",
        full_name="Cholic Acid 7-Sulfate",
        conjugation=Conjugation.SULFATED,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CA,
        notes="Sulfated at position 7",
    ),
    "CA-3-S": BileAcidSpecies(
        abbreviation="CA-3-S",
        full_name="Cholic Acid 3-Sulfate",
        conjugation=Conjugation.SULFATED,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CA,
        notes="Sulfated at position 3",
    ),
    "CDCA-3-S": BileAcidSpecies(
        abbreviation="CDCA-3-S",
        full_name="Chenodeoxycholic Acid 3-Sulfate",
        conjugation=Conjugation.SULFATED,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CDCA,
        notes="Sulfated at position 3",
    ),
    "DCA-3-S": BileAcidSpecies(
        abbreviation="DCA-3-S",
        full_name="Deoxycholic Acid 3-Sulfate",
        conjugation=Conjugation.SULFATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.DCA,
        notes="Sulfated at position 3",
    ),
    "LCA-3-S": BileAcidSpecies(
        abbreviation="LCA-3-S",
        full_name="Lithocholic Acid 3-Sulfate",
        conjugation=Conjugation.SULFATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.LCA,
        notes="Sulfated at position 3",
    ),
    "UDCA-3-S": BileAcidSpecies(
        abbreviation="UDCA-3-S",
        full_name="Ursodeoxycholic Acid 3-Sulfate",
        conjugation=Conjugation.SULFATED,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.UDCA,
        notes="Sulfated at position 3",
    ),
    
    # -------------------------------------------------------------------------
    # N-ACYL CONJUGATED BILE ACIDS
    # -------------------------------------------------------------------------
    "NCA": BileAcidSpecies(
        abbreviation="NCA",
        full_name="Norcholic Acid (N-acyl Cholic Acid)",
        conjugation=Conjugation.N_ACYL,
        origin=Origin.PRIMARY,
        core_structure=CoreStructure.CA,
        notes="N-acyl amidated form",
    ),
    "NDCA": BileAcidSpecies(
        abbreviation="NDCA",
        full_name="Nordeoxycholic Acid (N-acyl Deoxycholic Acid)",
        conjugation=Conjugation.N_ACYL,
        origin=Origin.SECONDARY,
        core_structure=CoreStructure.DCA,
        notes="N-acyl amidated form",
    ),
}


# =============================================================================
# HELPER FUNCTIONS FOR ACCESSING CLASSIFICATIONS
# =============================================================================

def get_species_by_conjugation(conjugation: Conjugation) -> List[str]:
    """Get all bile acid abbreviations with a specific conjugation type."""
    return [abbr for abbr, spec in BILE_ACID_PANEL.items() 
            if spec.conjugation == conjugation]


def get_species_by_origin(origin: Origin) -> List[str]:
    """Get all bile acid abbreviations with a specific origin."""
    return [abbr for abbr, spec in BILE_ACID_PANEL.items() 
            if spec.origin == origin]


def get_species_by_core(core: CoreStructure) -> List[str]:
    """Get all bile acid abbreviations derived from a specific core structure."""
    return [abbr for abbr, spec in BILE_ACID_PANEL.items() 
            if spec.core_structure == core]


def get_unconjugated() -> List[str]:
    """Get all unconjugated bile acids."""
    return get_species_by_conjugation(Conjugation.UNCONJUGATED)


def get_glycine_conjugated() -> List[str]:
    """Get all glycine-conjugated bile acids."""
    return get_species_by_conjugation(Conjugation.GLYCINE)


def get_taurine_conjugated() -> List[str]:
    """Get all taurine-conjugated bile acids."""
    return get_species_by_conjugation(Conjugation.TAURINE)


def get_sulfated() -> List[str]:
    """Get all sulfated bile acids."""
    return get_species_by_conjugation(Conjugation.SULFATED)


def get_conjugated() -> List[str]:
    """Get all conjugated bile acids (glycine + taurine + sulfated + n-acyl)."""
    return [abbr for abbr, spec in BILE_ACID_PANEL.items() 
            if spec.conjugation != Conjugation.UNCONJUGATED]


def get_primary() -> List[str]:
    """Get all primary bile acids."""
    return get_species_by_origin(Origin.PRIMARY)


def get_secondary() -> List[str]:
    """Get all secondary bile acids."""
    return get_species_by_origin(Origin.SECONDARY)


def get_keto_derivatives() -> List[str]:
    """Get all keto derivative bile acids."""
    return [abbr for abbr, spec in BILE_ACID_PANEL.items() 
            if spec.is_keto_derivative]


def get_all_species() -> List[str]:
    """Get list of all bile acid abbreviations in the panel."""
    return list(BILE_ACID_PANEL.keys())


def get_species_info(abbreviation: str) -> Optional[BileAcidSpecies]:
    """Get full information for a specific bile acid."""
    return BILE_ACID_PANEL.get(abbreviation)


def validate_columns(columns: List[str]) -> Dict[str, List[str]]:
    """
    Validate column names against the bile acid panel.
    
    Returns dict with 'matched', 'unmatched', and 'missing' lists.
    """
    panel_species = set(BILE_ACID_PANEL.keys())
    column_set = set(columns)
    
    matched = list(panel_species & column_set)
    unmatched = list(column_set - panel_species)
    missing = list(panel_species - column_set)
    
    return {
        "matched": matched,
        "unmatched": unmatched,  # columns not in panel (metadata or unknown species)
        "missing": missing,      # panel species not in columns
    }


# =============================================================================
# PREDEFINED ANALYSIS GROUPS
# =============================================================================
# These define common groupings used in bile acid analysis

ANALYSIS_GROUPS = {
    "total_primary": get_primary(),
    "total_secondary": get_secondary(),
    "total_conjugated": get_conjugated(),
    "total_unconjugated": get_unconjugated(),
    "glycine_conjugated": get_glycine_conjugated(),
    "taurine_conjugated": get_taurine_conjugated(),
    "sulfated": get_sulfated(),
    
    # CA family
    "all_CA": get_species_by_core(CoreStructure.CA),
    
    # CDCA family
    "all_CDCA": get_species_by_core(CoreStructure.CDCA),
    
    # DCA family
    "all_DCA": get_species_by_core(CoreStructure.DCA),
    
    # LCA family
    "all_LCA": get_species_by_core(CoreStructure.LCA),
    
    # UDCA family
    "all_UDCA": get_species_by_core(CoreStructure.UDCA),
    
    # Common clinical ratios (numerator species)
    "primary_unconjugated": [s for s in get_primary() if s in get_unconjugated()],
    "secondary_unconjugated": [s for s in get_secondary() if s in get_unconjugated()],
    "primary_conjugated": [s for s in get_primary() if s in get_conjugated()],
    "secondary_conjugated": [s for s in get_secondary() if s in get_conjugated()],
}


# =============================================================================
# CLINICAL RATIO DEFINITIONS
# =============================================================================

CLINICAL_RATIOS = {
    "TCA_to_GCA": {"numerator": ["TCA"], "denominator": ["GCA"]},
    "GCA_to_TCA": {"numerator": ["GCA"], "denominator": ["TCA"]},
    "GCDCA_to_TCDCA": {"numerator": ["GCDCA"], "denominator": ["TCDCA"]},
    "TCDCA_to_GCDCA": {"numerator": ["TCDCA"], "denominator": ["GCDCA"]},
    "CA_to_CDCA": {"numerator": ["CA"], "denominator": ["CDCA"]},
    "CDCA_to_CA": {"numerator": ["CDCA"], "denominator": ["CA"]},
    "CA_to_DCA": {"numerator": ["CA"], "denominator": ["DCA"]},
    "DCA_to_CA": {"numerator": ["DCA"], "denominator": ["CA"]},
    "primary_to_secondary": {
        "numerator": get_primary(), 
        "denominator": get_secondary()
    },
    "secondary_to_primary": {
        "numerator": get_secondary(), 
        "denominator": get_primary()
    },
    "glycine_to_taurine": {
        "numerator": get_glycine_conjugated(),
        "denominator": get_taurine_conjugated()
    },
    "taurine_to_glycine": {
        "numerator": get_taurine_conjugated(),
        "denominator": get_glycine_conjugated()
    },
    "conjugated_to_unconjugated": {
        "numerator": get_conjugated(),
        "denominator": get_unconjugated()
    },
    "unconjugated_to_conjugated": {
        "numerator": get_unconjugated(),
        "denominator": get_conjugated()
    },
    "12alpha_to_non12alpha": {
        # 12-alpha hydroxylated: CA, DCA and their conjugates
        "numerator": get_species_by_core(CoreStructure.CA) + get_species_by_core(CoreStructure.DCA),
        "denominator": [s for s in get_all_species() 
                       if BILE_ACID_PANEL[s].core_structure not in [CoreStructure.CA, CoreStructure.DCA]]
    },
}


if __name__ == "__main__":
    # Print summary of the panel
    print("=" * 60)
    print("BILE ACID PANEL SUMMARY")
    print("=" * 60)
    print(f"Total species in panel: {len(BILE_ACID_PANEL)}")
    print(f"\nBy conjugation:")
    print(f"  Unconjugated: {len(get_unconjugated())}")
    print(f"  Glycine-conjugated: {len(get_glycine_conjugated())}")
    print(f"  Taurine-conjugated: {len(get_taurine_conjugated())}")
    print(f"  Sulfated: {len(get_sulfated())}")
    print(f"\nBy origin:")
    print(f"  Primary: {len(get_primary())}")
    print(f"  Secondary: {len(get_secondary())}")
    print(f"\nKeto derivatives: {len(get_keto_derivatives())}")
