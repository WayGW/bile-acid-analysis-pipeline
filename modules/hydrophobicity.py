"""
Hydrophobicity Index Module for Bile Acid Analysis
====================================================

Calculates pH-dependent hydrophobicity indices using Henderson-Hasselbalch
ionization and Heuman (1989) reference values.

Formulas:
    f_protonated = 1 / (1 + 10^(pH - pKa))
    HI_effective = f_protonated * HI_protonated + (1 - f_protonated) * HI_ionized
    HI_pool = sum(fraction_i * HI_effective_i) per sample
    Hydrophobic_Load = total_concentration * HI_pool

References:
    Heuman DM. J Lipid Res. 1989;30(5):719-730.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.bile_acid_species import BILE_ACID_PANEL


class HydrophobicityCalculator:
    """
    Calculate pH-dependent hydrophobicity indices for bile acid data.

    At physiological pH (7.4), all bile acids are >99% ionized so
    HI_effective ≈ HI_ionized. The pH-dependent model becomes valuable
    for modeling intestinal environments (pH 5-7).
    """

    DEFAULT_PH = 7.4

    PH_ENVIRONMENTS = {
        "blood": 7.4,
        "duodenum": 6.0,
        "jejunum": 6.5,
        "ileum": 7.0,
        "cecum": 5.7,
        "colon": 6.5,
        "stomach": 2.0,
    }

    def __init__(self, pH: float = 7.4):
        if not 1.0 <= pH <= 14.0:
            raise ValueError(f"pH must be between 1.0 and 14.0, got {pH}")
        self.pH = pH
        self._species_cache = {
            abbr: (spec.pKa, spec.hi_ionized, spec.hi_protonated)
            for abbr, spec in BILE_ACID_PANEL.items()
        }

    @staticmethod
    def fraction_protonated(pH: float, pKa: float) -> float:
        """Henderson-Hasselbalch: fraction of acid in protonated (neutral) form."""
        return 1.0 / (1.0 + 10.0 ** (pH - pKa))

    def calculate_effective_hi(self, species_abbr: str) -> float:
        """Effective hydrophobicity index for a species at the current pH."""
        if species_abbr not in self._species_cache:
            raise ValueError(f"Unknown bile acid species: {species_abbr}")
        pKa, hi_ion, hi_prot = self._species_cache[species_abbr]
        f_prot = self.fraction_protonated(self.pH, pKa)
        return f_prot * hi_prot + (1.0 - f_prot) * hi_ion

    def calculate_all_effective_hi(self) -> Dict[str, float]:
        """Effective HI for all species in the panel at the current pH."""
        return {abbr: self.calculate_effective_hi(abbr) for abbr in self._species_cache}

    def calculate_protonation_table(self) -> Dict[str, Dict[str, float]]:
        """
        Full protonation and HI breakdown for all species.

        Returns dict of {abbr: {pKa, f_protonated, hi_ionized, hi_protonated, hi_effective}}.
        """
        table = {}
        for abbr, (pKa, hi_ion, hi_prot) in self._species_cache.items():
            f_prot = self.fraction_protonated(self.pH, pKa)
            table[abbr] = {
                "pKa": pKa,
                "f_protonated": f_prot,
                "f_ionized": 1.0 - f_prot,
                "hi_ionized": hi_ion,
                "hi_protonated": hi_prot,
                "hi_effective": f_prot * hi_prot + (1.0 - f_prot) * hi_ion,
            }
        return table

    def calculate_pool_hi(
        self, df: pd.DataFrame, bile_acid_cols: List[str]
    ) -> pd.Series:
        """
        Pool hydrophobicity index per sample (concentration-weighted average).

        Args:
            df: DataFrame with bile acid concentrations (rows=samples, cols=species).
            bile_acid_cols: Column names corresponding to bile acid species.

        Returns:
            Series of HI_pool values, one per sample.
        """
        available = [c for c in bile_acid_cols if c in df.columns and c in self._species_cache]
        if not available:
            return pd.Series(np.nan, index=df.index, name="HI_pool")

        hi_eff = np.array([self.calculate_effective_hi(c) for c in available])

        conc = df[available].values.astype(float)
        total = conc.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            fractions = conc / total[:, np.newaxis]
            fractions = np.where(np.isfinite(fractions), fractions, 0.0)

        pool_hi = fractions @ hi_eff
        pool_hi = np.where(total > 0, pool_hi, np.nan)

        return pd.Series(pool_hi, index=df.index, name="HI_pool")

    def calculate_hydrophobic_load(
        self, df: pd.DataFrame, bile_acid_cols: List[str]
    ) -> pd.Series:
        """
        Hydrophobic load = total_concentration x HI_pool.

        An extensive metric capturing both pool composition and total bile acid burden.
        """
        available = [c for c in bile_acid_cols if c in df.columns]
        total_conc = df[available].sum(axis=1) if available else pd.Series(0.0, index=df.index)
        pool_hi = self.calculate_pool_hi(df, bile_acid_cols)
        load = total_conc * pool_hi
        load.name = "hydrophobic_load"
        return load

    def calculate_pool_hi_multi_ph(
        self,
        df: pd.DataFrame,
        bile_acid_cols: List[str],
        ph_values: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Pool HI at multiple pH values for pH-sweep analysis.

        Returns DataFrame with columns HI_pool_pH_{value}.
        """
        if ph_values is None:
            ph_values = [2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.4, 8.0]

        results = pd.DataFrame(index=df.index)
        for ph in ph_values:
            calc = HydrophobicityCalculator(pH=ph)
            col_name = f"HI_pool_pH_{ph:.1f}"
            results[col_name] = calc.calculate_pool_hi(df, bile_acid_cols)
        return results

    def calculate_hydrophobicity_report(
        self, df: pd.DataFrame, bile_acid_cols: List[str]
    ) -> Dict:
        """
        Comprehensive hydrophobicity report.

        Returns:
            dict with keys: pool_hi, hydrophobic_load, species_hi_eff,
            protonation_table, ph, top_contributors.
        """
        available = [c for c in bile_acid_cols if c in df.columns and c in self._species_cache]
        pool_hi = self.calculate_pool_hi(df, bile_acid_cols)
        hydrophobic_load = self.calculate_hydrophobic_load(df, bile_acid_cols)
        species_hi = {c: self.calculate_effective_hi(c) for c in available}
        protonation = self.calculate_protonation_table()

        top_contributors = {}
        if available:
            conc = df[available].values.astype(float)
            total = conc.sum(axis=1)
            hi_eff = np.array([species_hi[c] for c in available])

            for idx in range(len(df)):
                if total[idx] <= 0:
                    top_contributors[df.index[idx]] = []
                    continue
                fracs = conc[idx] / total[idx]
                contributions = fracs * hi_eff
                ranked = sorted(
                    zip(available, contributions, fracs),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                top_contributors[df.index[idx]] = [
                    {"species": s, "contribution": float(c), "fraction": float(f)}
                    for s, c, f in ranked[:5]
                ]

        return {
            "pool_hi": pool_hi,
            "hydrophobic_load": hydrophobic_load,
            "species_hi_eff": species_hi,
            "protonation_table": protonation,
            "ph": self.pH,
            "top_contributors": top_contributors,
        }

    @staticmethod
    def ph_sweep_for_species(
        species_abbr: str,
        ph_range: Tuple[float, float] = (2.0, 9.0),
        n_points: int = 100,
    ) -> pd.DataFrame:
        """HI_effective vs pH curve for a single species."""
        if species_abbr not in BILE_ACID_PANEL:
            raise ValueError(f"Unknown bile acid species: {species_abbr}")

        spec = BILE_ACID_PANEL[species_abbr]
        ph_values = np.linspace(ph_range[0], ph_range[1], n_points)
        results = []
        for ph in ph_values:
            f_prot = HydrophobicityCalculator.fraction_protonated(ph, spec.pKa)
            hi_eff = f_prot * spec.hi_protonated + (1.0 - f_prot) * spec.hi_ionized
            results.append({
                "pH": ph,
                "f_protonated": f_prot,
                "hi_effective": hi_eff,
            })
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("HYDROPHOBICITY INDEX VERIFICATION")
    print("=" * 60)

    calc = HydrophobicityCalculator(pH=7.4)
    all_passed = True

    # Test 1: At pH = pKa, f_protonated should be exactly 0.5
    print("\n--- Test 1: Henderson-Hasselbalch at pH = pKa ---")
    f = calc.fraction_protonated(4.6, 4.6)
    status = "PASS" if abs(f - 0.5) < 1e-10 else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"  f_protonated(pH=4.6, pKa=4.6) = {f:.10f}  [{status}]")

    # Test 2: At pH 7.4, all species should be >99% ionized
    print("\n--- Test 2: All species >99% ionized at pH 7.4 ---")
    for abbr, (pKa, _, _) in calc._species_cache.items():
        f_prot = calc.fraction_protonated(7.4, pKa)
        if f_prot > 0.01:
            print(f"  WARNING: {abbr} is {f_prot*100:.2f}% protonated at pH 7.4")
    hi_eff = calc.calculate_all_effective_hi()
    max_diff = 0
    for abbr in calc._species_cache:
        hi_ion = calc._species_cache[abbr][1]
        diff = abs(hi_eff[abbr] - hi_ion)
        max_diff = max(max_diff, diff)
    status = "PASS" if max_diff < 0.01 else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"  Max |HI_eff - HI_ionized| at pH 7.4: {max_diff:.6f}  [{status}]")

    # Test 3: Rank order matches Heuman
    print("\n--- Test 3: Rank order (unconjugated) ---")
    unconj = ["LCA", "DCA", "CDCA", "CA", "UDCA"]
    unconj_hi = [hi_eff[s] for s in unconj]
    is_sorted = all(unconj_hi[i] >= unconj_hi[i + 1] for i in range(len(unconj_hi) - 1))
    status = "PASS" if is_sorted else "FAIL"
    if status == "FAIL":
        all_passed = False
    for s in unconj:
        print(f"  {s}: {hi_eff[s]:+.4f}")
    print(f"  Rank order LCA > DCA > CDCA > CA > UDCA: [{status}]")

    # Test 4: Single-species pool
    print("\n--- Test 4: Single-species pool ---")
    test_df = pd.DataFrame({"LCA": [100.0], "CA": [0.0]})
    pool = calc.calculate_pool_hi(test_df, ["LCA", "CA"])
    expected = hi_eff["LCA"]
    status = "PASS" if abs(pool.iloc[0] - expected) < 1e-10 else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"  100% LCA pool HI: {pool.iloc[0]:.4f}, expected: {expected:.4f}  [{status}]")

    # Test 5: Equal concentrations = arithmetic mean
    print("\n--- Test 5: Equal concentrations = arithmetic mean ---")
    species = list(calc._species_cache.keys())
    eq_data = {s: [10.0] for s in species}
    eq_df = pd.DataFrame(eq_data)
    pool_eq = calc.calculate_pool_hi(eq_df, species)
    mean_hi = np.mean([hi_eff[s] for s in species])
    status = "PASS" if abs(pool_eq.iloc[0] - mean_hi) < 1e-10 else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"  Equal-conc pool HI: {pool_eq.iloc[0]:.4f}, arithmetic mean: {mean_hi:.4f}  [{status}]")

    # Test 6: Hydrophobic Load
    print("\n--- Test 6: Hydrophobic Load ---")
    test_df2 = pd.DataFrame({"LCA": [200.0], "CA": [800.0]})
    pool2 = calc.calculate_pool_hi(test_df2, ["LCA", "CA"])
    load2 = calc.calculate_hydrophobic_load(test_df2, ["LCA", "CA"])
    expected_load = 1000.0 * pool2.iloc[0]
    status = "PASS" if abs(load2.iloc[0] - expected_load) < 1e-6 else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"  Total=1000, HI_pool={pool2.iloc[0]:.4f}, Load={load2.iloc[0]:.4f}, expected={expected_load:.4f}  [{status}]")

    # Test 7: Zero total concentration -> NaN
    print("\n--- Test 7: Zero concentration -> NaN ---")
    zero_df = pd.DataFrame({"LCA": [0.0], "CA": [0.0]})
    pool_zero = calc.calculate_pool_hi(zero_df, ["LCA", "CA"])
    status = "PASS" if np.isnan(pool_zero.iloc[0]) else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"  Zero-conc pool HI: {pool_zero.iloc[0]}  [{status}]")

    # Summary
    print("\n" + "=" * 60)
    overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"  {overall}")
    print("=" * 60)

    # Reference table
    print("\n--- Species Effective HI at pH 7.4 ---")
    sorted_species = sorted(hi_eff.items(), key=lambda x: x[1], reverse=True)
    for abbr, hi in sorted_species:
        pKa = calc._species_cache[abbr][0]
        f_prot = calc.fraction_protonated(7.4, pKa)
        print(f"  {abbr:20s}  HI_eff={hi:+.4f}  f_prot={f_prot:.6f}")
