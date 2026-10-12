"""
Main pipeline — ties the harness components together.

This script demonstrates the full data flow:
  raw data source -> parser -> store -> featurizer -> feature matrix

Run with:
    cd trna_prs
    python -m src.pipeline

For now this runs a demo with synthetic data.
Replace the synthetic section with real parser calls as data arrives.
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.featurizer import tRNAFeaturizer
from src.parsers.mitomap_parser import MitoMapParser
from src.schema.variant import (
    GenotypeData,
    PathogenicityClass,
    tRNAAnnotation,
    tRNARegion,
    UnifiedVariant,
    VariantAnnotation,
    VariantType,
)
from src.store.variant_store import VariantStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_synthetic_variants() -> list[UnifiedVariant]:
    """
    Build a set of synthetic variants that mimic real data structure.
    Used for pipeline testing before real cohort data is available.

    These are based on real known pathogenic mt-tRNA variants from MitoMap.
    """
    variants = [
        # MELAS — MT-TL1 A3243G, the most common mt disease mutation
        UnifiedVariant(
            variant_id="synthetic::chrM-3243-A-G",
            source="synthetic",
            variant_type=VariantType.MT_TRNA,
            chromosome="chrM",
            position=3243,
            reference_allele="A",
            alternate_allele="G",
            pathogenicity=PathogenicityClass.PATHOGENIC,
            disease_associations=["MELAS", "Diabetes mellitus", "CPEO"],
            trna_annotation=tRNAAnnotation(
                trna_gene="MT-TL1",
                isotype="Leu",
                anticodon="UAA",
                trna_region=tRNARegion.D_LOOP,
                is_wobble_position=False,
                is_anticodon_position=False,
                conservation_score=0.98,
            ),
            variant_annotation=VariantAnnotation(
                cadd_score=35.0,
                phylop_score=3.2,
                mitomap_status="Confirmed",
                gnomad_af=0.000012,
            ),
            genotype_data=GenotypeData(
                sample_id="SAMPLE_001",
                heteroplasmy_level=0.72,
                is_heteroplasmic=True,
                haplogroup="H1a",
            ),
        ),

        # MERRF — MT-TK A8344G
        UnifiedVariant(
            variant_id="synthetic::chrM-8344-A-G",
            source="synthetic",
            variant_type=VariantType.MT_TRNA,
            chromosome="chrM",
            position=8344,
            reference_allele="A",
            alternate_allele="G",
            pathogenicity=PathogenicityClass.PATHOGENIC,
            disease_associations=["MERRF", "Myoclonic epilepsy"],
            trna_annotation=tRNAAnnotation(
                trna_gene="MT-TK",
                isotype="Lys",
                anticodon="UUU",
                trna_region=tRNARegion.T_LOOP,
                is_wobble_position=False,
                is_anticodon_position=False,
                conservation_score=0.99,
            ),
            variant_annotation=VariantAnnotation(
                cadd_score=32.5,
                phylop_score=4.1,
                mitomap_status="Confirmed",
                gnomad_af=0.000008,
            ),
            genotype_data=GenotypeData(
                sample_id="SAMPLE_002",
                heteroplasmy_level=0.85,
                is_heteroplasmic=True,
                haplogroup="J1c",
            ),
        ),

        # Benign polymorphism — common in population
        UnifiedVariant(
            variant_id="synthetic::chrM-3290-T-C",
            source="synthetic",
            variant_type=VariantType.MT_TRNA,
            chromosome="chrM",
            position=3290,
            reference_allele="T",
            alternate_allele="C",
            pathogenicity=PathogenicityClass.BENIGN,
            disease_associations=[],
            trna_annotation=tRNAAnnotation(
                trna_gene="MT-TL1",
                isotype="Leu",
                anticodon="UAA",
                trna_region=tRNARegion.VARIABLE_LOOP,
                conservation_score=0.45,
            ),
            variant_annotation=VariantAnnotation(
                cadd_score=8.2,
                phylop_score=0.3,
                mitomap_status="Polymorphism",
                gnomad_af=0.0234,
            ),
            genotype_data=GenotypeData(
                sample_id="SAMPLE_003",
                heteroplasmy_level=1.0,
                is_heteroplasmic=False,
                haplogroup="U5b",
            ),
        ),

        # VUS in wobble position — uncertain significance
        UnifiedVariant(
            variant_id="synthetic::chrM-8320-T-C",
            source="synthetic",
            variant_type=VariantType.MT_TRNA,
            chromosome="chrM",
            position=8320,
            reference_allele="T",
            alternate_allele="C",
            pathogenicity=PathogenicityClass.VUS,
            disease_associations=["Possible mitochondrial myopathy"],
            trna_annotation=tRNAAnnotation(
                trna_gene="MT-TK",
                isotype="Lys",
                anticodon="UUU",
                trna_region=tRNARegion.WOBBLE_POSITION,
                is_wobble_position=True,
                is_anticodon_position=True,
                conservation_score=0.87,
            ),
            variant_annotation=VariantAnnotation(
                cadd_score=24.1,
                phylop_score=2.1,
                gnomad_af=None,  # Novel — not in gnomAD
            ),
            genotype_data=GenotypeData(
                sample_id="SAMPLE_004",
                heteroplasmy_level=0.23,
                is_heteroplasmic=True,
                haplogroup="K1a",
            ),
        ),

        # tRF (tRNA-derived fragment) example
        UnifiedVariant(
            variant_id="synthetic::chrM-8300-A-G-trf",
            source="synthetic",
            variant_type=VariantType.TRF,
            chromosome="chrM",
            position=8300,
            reference_allele="A",
            alternate_allele="G",
            pathogenicity=PathogenicityClass.UNKNOWN,
            disease_associations=[],
            trna_annotation=tRNAAnnotation(
                trna_gene="MT-TK",
                isotype="Lys",
                trf_type="5-tRF",
                trf_parent_gene="MT-TK",
                conservation_score=0.62,
            ),
            variant_annotation=VariantAnnotation(
                cadd_score=15.3,
                gnomad_af=0.0003,
            ),
        ),
    ]
    return variants


def run_pipeline(db_path: str = ":memory:") -> None:
    logger.info("=" * 60)
    logger.info("tRNA-PRS Data Harness — Pipeline Demo")
    logger.info("=" * 60)

    # 1. Initialize store
    logger.info("\n[1/4] Initializing variant store...")
    store = VariantStore(db_path)

    # 2. Load synthetic data (replace with real parser calls later)
    logger.info("\n[2/4] Loading variants into store...")
    synthetic_variants = build_synthetic_variants()
    inserted, skipped = store.insert_batch(synthetic_variants)
    logger.info(f"      Inserted: {inserted} | Skipped (duplicates): {skipped}")

    # Show how to use a real parser when data is available:
    # parser = MitoMapParser()
    # stats = store.ingest_from_parser(parser.parse_safe("data/raw/mitomap_disease.csv"))
    # logger.info(f"      MitoMap ingest: {stats}")

    # 3. Query and inspect
    logger.info("\n[3/4] Store statistics:")
    stats = store.stats()
    logger.info(f"      Total variants: {stats['total_variants']}")
    logger.info(f"      By type:         {stats['by_type']}")
    logger.info(f"      By pathogenicity:{stats['by_pathogenicity']}")

    # Example queries
    logger.info("\n      Example queries:")
    pathogenic = store.query(pathogenicity="pathogenic")
    logger.info(f"      Pathogenic variants: {len(pathogenic)}")
    wobble = store.query(is_wobble_position=True)
    logger.info(f"      Wobble position variants: {len(wobble)}")
    high_cadd = store.query(min_cadd=20.0)
    logger.info(f"      High CADD (>=20) variants: {len(high_cadd)}")

    # 4. Featurize
    logger.info("\n[4/4] Featurizing variants...")
    featurizer = tRNAFeaturizer()
    all_variants = store.query()
    feature_matrix = featurizer.to_matrix(all_variants)
    feature_names = featurizer.feature_names()

    logger.info(f"      Feature matrix shape: {feature_matrix.shape}")
    logger.info(f"      Feature dimensions:   {len(feature_names)}")
    logger.info(f"      Feature groups:       {featurizer.enabled_groups}")

    # Show a sample feature vector for the MELAS variant
    melas = store.get("synthetic::chrM-3243-A-G")
    if melas:
        melas_features = featurizer.featurize(melas)
        logger.info(f"\n      MELAS variant (A3243G) key features:")
        key_features = [
            "is_mitochondrial", "cadd_score", "heteroplasmy_level",
            "is_high_heteroplasmy", "mitomap_confirmed", "gnomad_af_log",
            "is_anticodon_loop", "trna_conservation_score",
        ]
        for feat in key_features:
            val = melas_features.get(feat, "N/A")
            logger.info(f"        {feat:<35} {val:.4f}" if isinstance(val, float) else f"        {feat:<35} {val}")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline demo complete.")
    logger.info("Next steps:")
    logger.info("  1. Download MitoMap data -> data/raw/mitomap_disease.csv")
    logger.info("  2. Download GtRNAdb BED -> data/raw/hg38-tRNAs.bed")
    logger.info("  3. Run: store.ingest_from_parser(MitoMapParser().parse_safe(...))")
    logger.info("  4. Add VCF data as cohort access is obtained")
    logger.info("  5. Feature matrix feeds directly into model training")
    logger.info("=" * 60)


if __name__ == "__main__":
    db_path = "data/db/variants.db"
    Path("data/db").mkdir(parents=True, exist_ok=True)
    run_pipeline(db_path)
