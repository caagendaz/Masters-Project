"""
VCF parser for WGS/WES cohort data.

Handles standard VCF 4.x files from:
  - UK Biobank (after data access approval)
  - Clinical WGS/WES pipelines (GATK, DeepVariant output)
  - Any other VCF-format source

Requires: pysam (wraps htslib for fast VCF/BCF parsing)
Install:  pip install pysam

Key design decisions:
  - Filters to tRNA gene regions only (using BED interval lookup)
  - Handles both germline VCFs (single sample) and multi-sample VCFs
  - Extracts heteroplasmy from MT allele fraction (AF field or AD calculation)
  - Skips low-quality calls based on configurable thresholds
"""

import logging
from pathlib import Path
from typing import Iterator, Optional

from src.parsers.base import BaseParser, ParserError
from src.parsers.mitomap_parser import MT_TRNA_GENE_MAP
from src.schema.variant import (
    GenotypeData,
    PathogenicityClass,
    Strand,
    tRNAAnnotation,
    tRNARegion,
    UnifiedVariant,
    VariantAnnotation,
    VariantType,
)

logger = logging.getLogger(__name__)

# Nuclear tRNA gene regions — abbreviated set from GtRNAdb hg38
# Full list should be loaded from a BED file in production
# Format: (chrom, start, end, gene_name, isotype, anticodon)
NUCLEAR_TRNA_REGIONS = [
    # These are representative examples; load full set from GtRNAdb BED
    ("chr1",  145334470, 145334542, "TRA-AGC1-1",  "Ala", "AGC"),
    ("chr1",  145367709, 145367781, "TRC-GCA1-1",  "Cys", "GCA"),
    ("chr6",  26553843,  26553915,  "TRE-CTC6-1",  "Glu", "CTC"),
    ("chr6",  26607438,  26607511,  "TRG-CCC6-1",  "Gly", "CCC"),
    # ... load rest from BED file
]

# Quality filter defaults — override via config
DEFAULT_FILTERS = {
    "min_gq": 20,          # Minimum genotype quality
    "min_dp": 10,          # Minimum read depth
    "min_af_mt": 0.01,     # Minimum allele fraction for mt variants (heteroplasmy threshold)
    "max_af_gnomad": 0.01, # Skip common variants (AF > 1% likely not pathogenic)
}


class VCFParser(BaseParser):
    """
    Parser for VCF files from WGS/WES pipelines.

    Usage:
        parser = VCFParser(config={"min_gq": 30, "sample_id": "UK123456"})
        for variant in parser.parse_safe("cohort.vcf.gz"):
            store.insert(variant)
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(source_name="vcf", config=config)
        self.filters = {**DEFAULT_FILTERS, **(config or {}).get("filters", {})}
        self._trna_intervals = self._build_interval_index()

    def validate_source(self, source: Path | str) -> bool:
        path = Path(source)
        if not path.exists():
            logger.error(f"VCF: file not found: {path}")
            return False
        suffix = path.suffix.lower()
        if suffix not in (".vcf", ".gz", ".bcf"):
            logger.warning(f"VCF: unexpected extension {suffix}, attempting anyway")
        return True

    def parse(self, source: Path | str) -> Iterator[UnifiedVariant]:
        try:
            import pysam
        except ImportError:
            raise ParserError(
                "pysam is required for VCF parsing. "
                "Install with: pip install pysam"
            )

        path = str(source)
        sample_id = self.config.get("sample_id", "unknown")

        with pysam.VariantFile(path) as vcf:
            # If multi-sample VCF, we can iterate per sample
            samples = list(vcf.header.samples)
            if sample_id != "unknown" and sample_id not in samples:
                logger.warning(f"VCF: sample {sample_id} not found, using first sample")
                sample_id = samples[0] if samples else "unknown"

            for record in vcf.fetch():
                # Filter to tRNA regions only
                trna_info = self._lookup_trna_region(record.chrom, record.start)
                if trna_info is None:
                    continue

                # Quality filters
                if not self._passes_quality_filters(record, sample_id):
                    self._skipped_count += 1
                    continue

                for alt in record.alts or []:
                    variant = self._build_variant(
                        record, alt, sample_id, trna_info
                    )
                    if variant:
                        yield variant

    def _build_variant(
        self, record, alt: str, sample_id: str, trna_info: dict
    ) -> Optional[UnifiedVariant]:
        """Convert a pysam variant record into a UnifiedVariant."""
        try:
            chrom = record.chrom
            pos = record.pos  # pysam returns 1-based pos
            ref = record.ref
            is_mitochondrial = chrom in ("chrM", "MT", "chrMT")

            variant_id = self._make_variant_id(chrom, pos, ref, alt)

            # Build tRNA annotation from lookup
            trna_annot = tRNAAnnotation(
                trna_gene=trna_info["gene"],
                isotype=trna_info.get("isotype"),
                anticodon=trna_info.get("anticodon"),
                trna_region=tRNARegion.UNKNOWN,  # Would be refined by SPRINZL lookup
            )

            # Build genotype data
            genotype_data = self._extract_genotype(
                record, sample_id, is_mitochondrial
            )

            # Build variant annotation from INFO fields
            variant_annot = self._extract_info_annotations(record)

            return UnifiedVariant(
                variant_id=variant_id,
                source=self.source_name,
                variant_type=(
                    VariantType.MT_TRNA if is_mitochondrial
                    else VariantType.NUCLEAR_TRNA
                ),
                chromosome=chrom,
                position=pos,
                reference_allele=ref,
                alternate_allele=alt,
                strand=Strand.PLUS,
                pathogenicity=PathogenicityClass.UNKNOWN,  # Annotated downstream
                trna_annotation=trna_annot,
                variant_annotation=variant_annot,
                genotype_data=genotype_data,
                raw_record={
                    "chrom": chrom, "pos": pos,
                    "ref": ref, "alt": alt,
                    "filter": list(record.filter.keys()),
                },
            )

        except Exception as e:
            logger.debug(f"VCF: failed to build variant at {record.chrom}:{record.pos}: {e}")
            self._error_count += 1
            return None

    def _extract_genotype(
        self, record, sample_id: str, is_mitochondrial: bool
    ) -> GenotypeData:
        """Extract sample-level genotype fields."""
        gd = GenotypeData(sample_id=sample_id)
        try:
            if sample_id in record.samples:
                sample = record.samples[sample_id]
                gt = sample.get("GT")
                if gt:
                    gd.genotype = "/".join(str(a) for a in gt)
                gd.genotype_quality = sample.get("GQ")
                gd.read_depth = sample.get("DP")

                ad = sample.get("AD")
                if ad and len(ad) >= 2:
                    gd.allele_depth = ",".join(str(x) for x in ad)

                # Heteroplasmy for mt-DNA variants
                if is_mitochondrial:
                    af = sample.get("AF") or record.info.get("AF")
                    if af is not None:
                        af_val = af[0] if isinstance(af, (list, tuple)) else af
                        gd.heteroplasmy_level = float(af_val)
                        gd.is_heteroplasmic = (
                            0.01 < gd.heteroplasmy_level < 0.99
                        )
                    # Calculate from AD if AF not available
                    elif ad and len(ad) >= 2 and sum(ad) > 0:
                        gd.heteroplasmy_level = ad[1] / sum(ad)
                        gd.is_heteroplasmic = (
                            0.01 < gd.heteroplasmy_level < 0.99
                        )

        except Exception as e:
            logger.debug(f"VCF: genotype extraction issue: {e}")

        return gd

    def _extract_info_annotations(self, record) -> VariantAnnotation:
        """Extract standard INFO field annotations if present."""
        va = VariantAnnotation()
        try:
            info = record.info
            # CADD scores (if pre-annotated with CADD)
            cadd = info.get("CADD_PHRED") or info.get("CADD")
            if cadd is not None:
                va.cadd_score = float(cadd[0] if isinstance(cadd, tuple) else cadd)

            # gnomAD allele frequency (if pre-annotated)
            af = info.get("gnomAD_AF") or info.get("AF_gnomad")
            if af is not None:
                va.gnomad_af = float(af[0] if isinstance(af, tuple) else af)

            # dbSNP rsID
            if record.id:
                va.dbsnp_id = record.id

        except Exception as e:
            logger.debug(f"VCF: INFO extraction issue: {e}")

        return va

    def _passes_quality_filters(self, record, sample_id: str) -> bool:
        """Apply configurable quality filters."""
        try:
            if sample_id in record.samples:
                sample = record.samples[sample_id]
                gq = sample.get("GQ")
                if gq is not None and gq < self.filters["min_gq"]:
                    return False
                dp = sample.get("DP")
                if dp is not None and dp < self.filters["min_dp"]:
                    return False
        except Exception:
            pass
        return True

    def _build_interval_index(self) -> list:
        """
        Build a simple interval list for tRNA region lookups.
        In production, replace with an interval tree (e.g. ncls or intervaltree package)
        for O(log n) lookup performance on large VCFs.
        """
        intervals = []

        # Add mt-tRNA regions
        for gene, (start, end, strand) in MT_TRNA_GENE_MAP.items():
            intervals.append({
                "chrom": "chrM",
                "start": start,
                "end": end,
                "gene": gene,
                "strand": strand,
                "isotype": None,
                "anticodon": None,
            })

        # Add nuclear tRNA regions
        for chrom, start, end, gene, isotype, anticodon in NUCLEAR_TRNA_REGIONS:
            intervals.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "gene": gene,
                "isotype": isotype,
                "anticodon": anticodon,
                "strand": "+",
            })

        return intervals

    def _lookup_trna_region(
        self, chrom: str, pos: int
    ) -> Optional[dict]:
        """Return tRNA region info if position overlaps a tRNA gene, else None."""
        # Normalize chromosome names
        chrom_norm = chrom if chrom.startswith("chr") else f"chr{chrom}"
        if chrom_norm in ("chrMT", "chrM"):
            chrom_norm = "chrM"

        for interval in self._trna_intervals:
            if (interval["chrom"] == chrom_norm
                    and interval["start"] <= pos <= interval["end"]):
                return interval
        return None

    def load_trna_bed(self, bed_path: Path) -> None:
        """
        Load a full tRNA gene BED file from GtRNAdb to replace the
        hardcoded NUCLEAR_TRNA_REGIONS above.

        Download from: http://gtrnadb.ucsc.edu/genomes/eukaryotes/Hsapi38/
        File: hg38-tRNAs.bed
        """
        self._trna_intervals = [
            i for i in self._trna_intervals if i["chrom"] == "chrM"
        ]

        with open(bed_path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 6:
                    continue
                chrom, start, end, name, score, strand = parts[:6]
                # GtRNAdb name format: chr6.tRNA1-AlaAGC
                isotype, anticodon = self._parse_gtrnadb_name(name)
                self._trna_intervals.append({
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "gene": name,
                    "strand": strand,
                    "isotype": isotype,
                    "anticodon": anticodon,
                })
        logger.info(f"VCF: loaded {len(self._trna_intervals)} tRNA regions from BED")

    @staticmethod
    def _parse_gtrnadb_name(name: str) -> tuple[Optional[str], Optional[str]]:
        """Parse GtRNAdb name e.g. 'chr6.tRNA1-AlaAGC' -> ('Ala', 'AGC')"""
        try:
            suffix = name.split("-", 1)[1]  # 'AlaAGC'
            isotype = suffix[:3]
            anticodon = suffix[3:6]
            return isotype, anticodon
        except (IndexError, ValueError):
            return None, None
