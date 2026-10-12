"""
Unified variant schema for tRNA-PRS project.
Every data source parser normalizes its output into these dataclasses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VariantType(Enum):
    MT_TRNA = "mt_trna"           # Mitochondrial tRNA variant
    NUCLEAR_TRNA = "nuclear_trna" # Nuclear-encoded tRNA variant
    TRF = "trf"                   # tRNA-derived fragment
    UNKNOWN = "unknown"


class Strand(Enum):
    PLUS = "+"
    MINUS = "-"
    UNKNOWN = "."


class PathogenicityClass(Enum):
    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    VUS = "vus"                   # Variant of uncertain significance
    LIKELY_BENIGN = "likely_benign"
    BENIGN = "benign"
    UNKNOWN = "unknown"


class tRNARegion(Enum):
    """Structural regions of the tRNA molecule."""
    ACCEPTOR_STEM = "acceptor_stem"       # 1-7 / 66-72, amino acid attachment
    D_LOOP = "d_loop"                     # 14-25, contains dihydrouridine
    D_STEM = "d_stem"                     # 10-13 / 22-25
    ANTICODON_STEM = "anticodon_stem"     # 27-31 / 39-43
    ANTICODON_LOOP = "anticodon_loop"     # 32-38, positions 34/35/36 decode codon
    WOBBLE_POSITION = "wobble_position"   # position 34 specifically
    VARIABLE_LOOP = "variable_loop"       # 44-48, length varies by tRNA type
    T_STEM = "t_stem"                     # 49-53 / 61-65
    T_LOOP = "t_loop"                     # 54-60
    CCA_TAIL = "cca_tail"                 # 73-75, amino acid attachment site
    UNKNOWN = "unknown"


@dataclass
class tRNAAnnotation:
    """
    tRNA-specific structural and functional annotations.
    These go beyond standard genomic variant annotations.
    """
    trna_gene: Optional[str] = None          # e.g. "MT-TL1", "TRA-AGC1-1"
    isotype: Optional[str] = None            # amino acid carried e.g. "Leu", "Ala"
    anticodon: Optional[str] = None          # e.g. "UAA", "GGC"
    trna_region: Optional[tRNARegion] = None
    position_in_trna: Optional[int] = None   # Sprinzl coordinate system position
    is_wobble_position: bool = False         # position 34 flag
    is_anticodon_position: bool = False      # positions 34/35/36
    modification_site: Optional[str] = None  # known modification at this position
    conservation_score: Optional[float] = None  # phylogenetic conservation
    structural_impact: Optional[str] = None  # predicted effect on tRNA fold
    # tRF-specific fields
    trf_type: Optional[str] = None           # "5-tRF", "3-tRF", "i-tRF", "tiRNA"
    trf_parent_gene: Optional[str] = None    # parent tRNA this fragment derives from


@dataclass
class VariantAnnotation:
    """
    Standard genomic variant annotations aggregated from external tools/databases.
    """
    cadd_score: Optional[float] = None       # CADD phred score (>20 = top 1% deleterious)
    phylop_score: Optional[float] = None     # Conservation: positive = conserved
    phastcons_score: Optional[float] = None  # Conservation: 0-1
    sift_score: Optional[float] = None       # Functional effect (for coding variants)
    clinvar_id: Optional[str] = None
    clinvar_significance: Optional[str] = None
    mitomap_status: Optional[str] = None     # MitoMap disease association status
    gnomad_af: Optional[float] = None        # gnomAD allele frequency (population)
    gnomad_af_popmax: Optional[float] = None # Highest AF across gnomAD populations
    dbsnp_id: Optional[str] = None


@dataclass
class GenotypeData:
    """
    Sample-level genotype information.
    Kept separate from variant-level data for clean schema design.
    """
    sample_id: Optional[str] = None
    genotype: Optional[str] = None           # e.g. "0/1", "1/1"
    allele_depth: Optional[str] = None       # e.g. "15,20"
    genotype_quality: Optional[int] = None
    read_depth: Optional[int] = None
    # mt-DNA specific
    heteroplasmy_level: Optional[float] = None  # 0.0-1.0, fraction of mt copies with variant
    is_heteroplasmic: Optional[bool] = None
    haplogroup: Optional[str] = None            # mitochondrial haplogroup e.g. "H1a"


@dataclass
class UnifiedVariant:
    """
    The central data model. Every parser produces UnifiedVariant objects.
    This is what gets stored, featurized, and fed to the model.
    """
    # --- Identity ---
    variant_id: str                           # Internal unique ID
    source: str                               # Which parser/database produced this
    variant_type: VariantType = VariantType.UNKNOWN

    # --- Genomic coordinates (GRCh38 / hg38) ---
    chromosome: Optional[str] = None          # "chrM", "chr1", etc.
    position: Optional[int] = None            # 1-based
    reference_allele: Optional[str] = None
    alternate_allele: Optional[str] = None
    strand: Strand = Strand.UNKNOWN

    # --- Classification ---
    pathogenicity: PathogenicityClass = PathogenicityClass.UNKNOWN
    disease_associations: list[str] = field(default_factory=list)

    # --- Layered annotations ---
    trna_annotation: Optional[tRNAAnnotation] = None
    variant_annotation: Optional[VariantAnnotation] = None
    genotype_data: Optional[GenotypeData] = None

    # --- Provenance ---
    raw_record: Optional[dict] = None         # Original record before normalization

    def is_anticodon_variant(self) -> bool:
        """Convenience: is this variant in the anticodon loop?"""
        if self.trna_annotation is None:
            return False
        return self.trna_annotation.trna_region in (
            tRNARegion.ANTICODON_LOOP,
            tRNARegion.WOBBLE_POSITION,
        )

    def is_high_confidence_pathogenic(self) -> bool:
        """Convenience: pathogenic or likely pathogenic with CADD support."""
        is_path = self.pathogenicity in (
            PathogenicityClass.PATHOGENIC,
            PathogenicityClass.LIKELY_PATHOGENIC,
        )
        cadd = (
            self.variant_annotation.cadd_score
            if self.variant_annotation else None
        )
        return is_path and (cadd is None or cadd >= 20.0)

    def __repr__(self):
        return (
            f"UnifiedVariant({self.variant_id} | "
            f"{self.chromosome}:{self.position} "
            f"{self.reference_allele}>{self.alternate_allele} | "
            f"{self.variant_type.value} | "
            f"{self.pathogenicity.value})"
        )
