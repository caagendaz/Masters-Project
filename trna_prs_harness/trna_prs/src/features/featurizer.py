"""
Feature engineering layer.

Converts UnifiedVariant objects into numerical feature vectors
suitable for ML models. This sits between the data store and
the model training pipeline.

Design:
  - Each feature group is a separate method (easy to ablate)
  - Returns a dict first, then converts to numpy/pandas
  - Missing values handled explicitly (not silently dropped)
  - tRNA-specific features are first-class, not afterthoughts
"""

import logging
from typing import Optional

import numpy as np

from src.schema.variant import tRNARegion, UnifiedVariant, VariantType

logger = logging.getLogger(__name__)

# Default fill values for missing features
MISSING_FLOAT = -1.0   # Sentinel for missing continuous features
MISSING_INT = 0        # For binary/categorical flags


class tRNAFeaturizer:
    """
    Converts UnifiedVariant objects into fixed-length feature vectors.

    Feature groups (each can be toggled independently for ablation):
      1. genomic_position    - chromosome, position, variant type encoding
      2. trna_structural     - tRNA region, wobble position, anticodon flags
      3. conservation        - phyloP, phastCons, cross-species conservation
      4. pathogenicity_prior - CADD, ClinVar, MitoMap status
      5. population          - gnomAD AF (rarity = signal)
      6. heteroplasmy        - mt-DNA specific, heteroplasmy level & flag
      7. isotype_encoding    - which amino acid this tRNA carries (one-hot)
      8. region_encoding     - structural region one-hot

    Usage:
        featurizer = tRNAFeaturizer()
        features = featurizer.featurize(variant)    # -> dict
        vector = featurizer.to_vector(variant)      # -> np.ndarray
        matrix = featurizer.to_matrix(variant_list) # -> np.ndarray (N x D)
    """

    # All 20 standard amino acid isotypes for one-hot encoding
    ISOTYPES = [
        "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly",
        "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser",
        "Thr", "Trp", "Tyr", "Val",
    ]

    TRNA_REGIONS = [r.value for r in tRNARegion if r != tRNARegion.UNKNOWN]

    VARIANT_TYPES = [vt.value for vt in VariantType if vt != VariantType.UNKNOWN]

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.enabled_groups = self.config.get("feature_groups", [
            "genomic_position",
            "trna_structural",
            "conservation",
            "pathogenicity_prior",
            "population",
            "heteroplasmy",
            "isotype_encoding",
            "region_encoding",
        ])

    def featurize(self, variant: UnifiedVariant) -> dict:
        """
        Convert a single variant to a feature dictionary.
        All values are floats; missing values use sentinel constants.
        """
        features = {}

        if "genomic_position" in self.enabled_groups:
            features.update(self._genomic_position_features(variant))

        if "trna_structural" in self.enabled_groups:
            features.update(self._trna_structural_features(variant))

        if "conservation" in self.enabled_groups:
            features.update(self._conservation_features(variant))

        if "pathogenicity_prior" in self.enabled_groups:
            features.update(self._pathogenicity_prior_features(variant))

        if "population" in self.enabled_groups:
            features.update(self._population_features(variant))

        if "heteroplasmy" in self.enabled_groups:
            features.update(self._heteroplasmy_features(variant))

        if "isotype_encoding" in self.enabled_groups:
            features.update(self._isotype_one_hot(variant))

        if "region_encoding" in self.enabled_groups:
            features.update(self._region_one_hot(variant))

        return features

    def to_vector(self, variant: UnifiedVariant) -> np.ndarray:
        """Convert a variant to a 1D numpy array. Feature order is deterministic."""
        features = self.featurize(variant)
        return np.array(list(features.values()), dtype=np.float32)

    def to_matrix(self, variants: list[UnifiedVariant]) -> np.ndarray:
        """Convert a list of variants to a 2D feature matrix (N x D)."""
        if not variants:
            return np.empty((0, 0))
        rows = [self.to_vector(v) for v in variants]
        return np.stack(rows, axis=0)

    def feature_names(self) -> list[str]:
        """Return the ordered list of feature names (matches to_vector output)."""
        dummy = UnifiedVariant(variant_id="dummy", source="dummy")
        return list(self.featurize(dummy).keys())

    # ------------------------------------------------------------------ #
    # Feature group implementations
    # ------------------------------------------------------------------ #

    def _genomic_position_features(self, v: UnifiedVariant) -> dict:
        return {
            # Is this a mitochondrial variant? Strong prior signal
            "is_mitochondrial": float(v.variant_type == VariantType.MT_TRNA),
            "is_nuclear_trna":  float(v.variant_type == VariantType.NUCLEAR_TRNA),
            "is_trf":           float(v.variant_type == VariantType.TRF),
            # Transition vs transversion (ts/tv ratio is biologically meaningful)
            "is_transition":    float(self._is_transition(
                v.reference_allele, v.alternate_allele
            )),
        }

    def _trna_structural_features(self, v: UnifiedVariant) -> dict:
        ta = v.trna_annotation
        if ta is None:
            return {
                "is_wobble_position":   0.0,
                "is_anticodon_position": 0.0,
                "is_acceptor_stem":     0.0,
                "is_anticodon_loop":    0.0,
                "has_known_modification": 0.0,
                "position_in_trna":     MISSING_FLOAT,
            }
        return {
            # These are among the most important features biologically
            "is_wobble_position":    float(ta.is_wobble_position),
            "is_anticodon_position": float(ta.is_anticodon_position),
            "is_acceptor_stem": float(
                ta.trna_region == tRNARegion.ACCEPTOR_STEM
            ),
            "is_anticodon_loop": float(
                ta.trna_region in (tRNARegion.ANTICODON_LOOP, tRNARegion.WOBBLE_POSITION)
            ),
            "has_known_modification": float(ta.modification_site is not None),
            "position_in_trna": float(ta.position_in_trna)
                if ta.position_in_trna is not None else MISSING_FLOAT,
        }

    def _conservation_features(self, v: UnifiedVariant) -> dict:
        ta = v.trna_annotation
        va = v.variant_annotation
        return {
            "phylop_score": float(va.phylop_score)
                if va and va.phylop_score is not None else MISSING_FLOAT,
            "phastcons_score": float(va.phastcons_score)
                if va and va.phastcons_score is not None else MISSING_FLOAT,
            "trna_conservation_score": float(ta.conservation_score)
                if ta and ta.conservation_score is not None else MISSING_FLOAT,
        }

    def _pathogenicity_prior_features(self, v: UnifiedVariant) -> dict:
        va = v.variant_annotation
        return {
            # CADD phred score — log-scaled deleteriousness
            "cadd_score": float(va.cadd_score)
                if va and va.cadd_score is not None else MISSING_FLOAT,
            # Binary: is this in ClinVar at all?
            "in_clinvar": float(va is not None and va.clinvar_id is not None),
            # Binary: MitoMap confirmed pathogenic
            "mitomap_confirmed": float(
                va is not None
                and va.mitomap_status is not None
                and "confirmed" in str(va.mitomap_status).lower()
            ),
            # Binary: has disease associations in our store
            "has_disease_association": float(bool(v.disease_associations)),
            "n_disease_associations": float(len(v.disease_associations)),
        }

    def _population_features(self, v: UnifiedVariant) -> dict:
        va = v.variant_annotation
        gnomad_af = va.gnomad_af if va and va.gnomad_af is not None else None

        return {
            # Log-transform AF: rare variants (small AF) get larger negative values
            # Missing AF treated as very rare (assume not in gnomAD = rare/novel)
            "gnomad_af_log": float(np.log10(gnomad_af + 1e-10))
                if gnomad_af is not None else float(np.log10(1e-10)),
            "gnomad_af_raw": float(gnomad_af) if gnomad_af is not None else MISSING_FLOAT,
            "is_novel_variant": float(gnomad_af is None),    # Not in gnomAD at all
            "is_rare": float(gnomad_af is not None and gnomad_af < 0.001),
            "is_common": float(gnomad_af is not None and gnomad_af > 0.01),
        }

    def _heteroplasmy_features(self, v: UnifiedVariant) -> dict:
        gd = v.genotype_data
        if gd is None or gd.heteroplasmy_level is None:
            return {
                "heteroplasmy_level": MISSING_FLOAT,
                "is_heteroplasmic": 0.0,
                "is_homoplasmic": 0.0,
                "is_high_heteroplasmy": 0.0,  # >50% — more likely to cause disease
            }
        hl = gd.heteroplasmy_level
        return {
            "heteroplasmy_level": float(hl),
            "is_heteroplasmic": float(0.01 < hl < 0.99),
            "is_homoplasmic": float(hl >= 0.99),
            "is_high_heteroplasmy": float(hl > 0.5),
        }

    def _isotype_one_hot(self, v: UnifiedVariant) -> dict:
        """One-hot encode the amino acid isotype the tRNA carries."""
        ta = v.trna_annotation
        isotype = ta.isotype if ta else None
        return {
            f"isotype_{iso}": float(isotype == iso)
            for iso in self.ISOTYPES
        }

    def _region_one_hot(self, v: UnifiedVariant) -> dict:
        """One-hot encode the structural region of the tRNA."""
        ta = v.trna_annotation
        region = ta.trna_region.value if (ta and ta.trna_region) else None
        return {
            f"region_{reg}": float(region == reg)
            for reg in self.TRNA_REGIONS
        }

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_transition(ref: Optional[str], alt: Optional[str]) -> bool:
        """
        Transitions (A<->G, C<->T) are more common and often less damaging
        than transversions. Useful prior for pathogenicity.
        """
        if not ref or not alt:
            return False
        transitions = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
        return (ref.upper(), alt.upper()) in transitions
