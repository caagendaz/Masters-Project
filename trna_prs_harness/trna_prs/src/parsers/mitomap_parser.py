"""
MitoMap parser.

MitoMap (mitomap.org) is the primary curated database for human
mitochondrial DNA variants. It covers mt-tRNA variants with disease
associations, pathogenicity calls, and literature evidence.

Data can be downloaded as JSON or CSV from:
https://mitomap.org/foswiki/bin/view/MITOMAP/Resources

Expected input: MitoMap disease mutations CSV/JSON export.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from src.parsers.base import BaseParser
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

# MitoMap mt-tRNA gene position ranges on the mitochondrial genome (rCRS)
# Format: gene_name -> (start, end, strand)
MT_TRNA_GENE_MAP = {
    "MT-TF":  (577,  647,  "+"),   # Phe
    "MT-TV":  (1602, 1670, "+"),   # Val
    "MT-TL1": (3230, 3304, "+"),   # Leu (UUA/UUG) — most commonly mutated
    "MT-TI":  (4263, 4331, "+"),   # Ile
    "MT-TQ":  (4329, 4400, "-"),   # Gln
    "MT-TM":  (4402, 4469, "+"),   # Met
    "MT-TW":  (5512, 5579, "+"),   # Trp
    "MT-TA":  (5587, 5655, "-"),   # Ala
    "MT-TN":  (5657, 5729, "-"),   # Asn
    "MT-TC":  (5761, 5826, "-"),   # Cys
    "MT-TY":  (5826, 5891, "-"),   # Tyr
    "MT-TS1": (7446, 7514, "-"),   # Ser (UCN)
    "MT-TD":  (7518, 7585, "+"),   # Asp
    "MT-TK":  (8295, 8364, "+"),   # Lys — MERRF mutations here
    "MT-TG":  (9991, 10058,"+"),   # Gly
    "MT-TR":  (10405,10469,"+"),   # Arg
    "MT-TH":  (12138,12206,"+"),   # His
    "MT-TS2": (12207,12265,"+"),   # Ser (AGU/AGC)
    "MT-TL2": (12266,12336,"+"),   # Leu (CUN)
    "MT-TE":  (14674,14742,"-"),   # Glu
    "MT-TT":  (15888,15953,"+"),   # Thr
    "MT-TP":  (15956,16023,"-"),   # Pro
}

# Known high-impact positions within mt-tRNA genes
# These warrant special flags in the annotation
KNOWN_PATHOGENIC_POSITIONS = {
    3243,   # MT-TL1 A3243G — MELAS, most common mt disease mutation
    8344,   # MT-TK A8344G — MERRF
    8356,   # MT-TK T8356C
    3271,   # MT-TL1 T3271C
    3291,   # MT-TL1 T3291C
}

# Mapping MitoMap status strings to our PathogenicityClass enum
MITOMAP_PATHOGENICITY_MAP = {
    "confirmed":         PathogenicityClass.PATHOGENIC,
    "reported":          PathogenicityClass.LIKELY_PATHOGENIC,
    "polymorphism":      PathogenicityClass.BENIGN,
    "polymorphism/":     PathogenicityClass.BENIGN,
    "conflicting":       PathogenicityClass.VUS,
    "":                  PathogenicityClass.UNKNOWN,
}


class MitoMapParser(BaseParser):
    """
    Parser for MitoMap disease mutations data.

    Handles both JSON (API response) and CSV (download export) formats.
    Auto-detects format from file extension.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(source_name="mitomap", config=config)

    def validate_source(self, source: Path | str) -> bool:
        path = Path(source)
        if not path.exists():
            logger.error(f"MitoMap: file not found: {path}")
            return False
        if path.suffix.lower() not in (".json", ".csv", ".tsv"):
            logger.warning(f"MitoMap: unexpected file extension {path.suffix}")
        return True

    def parse(self, source: Path | str) -> Iterator[UnifiedVariant]:
        path = Path(source)
        if path.suffix.lower() == ".json":
            yield from self._parse_json(path)
        else:
            yield from self._parse_csv(path)

    def _parse_json(self, path: Path) -> Iterator[UnifiedVariant]:
        with open(path) as f:
            data = json.load(f)

        # MitoMap JSON can be nested under a "data" key
        records = data if isinstance(data, list) else data.get("data", [])
        for record in records:
            variant = self._normalize_record(record)
            if variant:
                yield variant

    def _parse_csv(self, path: Path) -> Iterator[UnifiedVariant]:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                variant = self._normalize_record(dict(row))
                if variant:
                    yield variant

    def _normalize_record(self, record: dict) -> Optional[UnifiedVariant]:
        """Convert a raw MitoMap record into a UnifiedVariant."""
        try:
            # MitoMap uses 'Position' for mt genome coordinate
            pos_raw = record.get("Position") or record.get("position")
            if pos_raw is None:
                self._skipped_count += 1
                return None
            position = int(str(pos_raw).replace(",", ""))

            ref = str(record.get("Ref", record.get("ref", "N"))).upper().strip()
            alt = str(record.get("Alt", record.get("alt", "N"))).upper().strip()

            variant_id = self._make_variant_id("chrM", position, ref, alt)

            # Determine which tRNA gene this position falls in
            trna_gene = self._get_trna_gene(position)
            if trna_gene is None:
                self._skipped_count += 1
                return None  # Not in a tRNA gene

            # Build tRNA annotation
            trna_annot = tRNAAnnotation(
                trna_gene=trna_gene,
                isotype=self._get_isotype(trna_gene),
                trna_region=self._infer_region(position, trna_gene),
                is_wobble_position=self._is_wobble_position(position, trna_gene),
                is_anticodon_position=self._is_anticodon_position(position, trna_gene),
            )

            # Build variant annotation
            status_raw = str(record.get("Disease Status", record.get("status", ""))).lower()
            pathogenicity = MITOMAP_PATHOGENICITY_MAP.get(
                status_raw, PathogenicityClass.UNKNOWN
            )

            # Disease associations — MitoMap often has these as semicolon-separated
            diseases_raw = record.get("Disease", record.get("disease", ""))
            diseases = (
                [d.strip() for d in str(diseases_raw).split(";") if d.strip()]
                if diseases_raw else []
            )

            variant_annot = VariantAnnotation(
                mitomap_status=str(record.get("Disease Status", "")),
                clinvar_id=record.get("ClinVar", None),
            )

            return UnifiedVariant(
                variant_id=variant_id,
                source=self.source_name,
                variant_type=VariantType.MT_TRNA,
                chromosome="chrM",
                position=position,
                reference_allele=ref,
                alternate_allele=alt,
                strand=Strand.PLUS,  # mt genome is always reported on + strand
                pathogenicity=pathogenicity,
                disease_associations=diseases,
                trna_annotation=trna_annot,
                variant_annotation=variant_annot,
                raw_record=record,
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"MitoMap: could not parse record {record}: {e}")
            self._error_count += 1
            return None

    def _get_trna_gene(self, position: int) -> Optional[str]:
        """Return tRNA gene name if position falls within a known mt-tRNA gene."""
        for gene, (start, end, _) in MT_TRNA_GENE_MAP.items():
            if start <= position <= end:
                return gene
        return None

    def _get_isotype(self, gene: str) -> Optional[str]:
        """Extract amino acid isotype from gene name e.g. MT-TL1 -> Leu."""
        isotype_map = {
            "MT-TF": "Phe", "MT-TV": "Val", "MT-TL1": "Leu", "MT-TI": "Ile",
            "MT-TQ": "Gln", "MT-TM": "Met", "MT-TW": "Trp", "MT-TA": "Ala",
            "MT-TN": "Asn", "MT-TC": "Cys", "MT-TY": "Tyr", "MT-TS1": "Ser",
            "MT-TD": "Asp", "MT-TK": "Lys", "MT-TG": "Gly", "MT-TR": "Arg",
            "MT-TH": "His", "MT-TS2": "Ser", "MT-TL2": "Leu", "MT-TE": "Glu",
            "MT-TT": "Thr", "MT-TP": "Pro",
        }
        return isotype_map.get(gene)

    def _infer_region(self, position: int, gene: str) -> tRNARegion:
        """
        Rough structural region inference based on relative position within gene.
        A proper implementation would use the SPRINZL coordinate database.
        This is a placeholder that works for basic stratification.
        """
        if gene not in MT_TRNA_GENE_MAP:
            return tRNARegion.UNKNOWN

        start, end, _ = MT_TRNA_GENE_MAP[gene]
        gene_length = end - start
        relative_pos = (position - start) / gene_length

        # Rough structural mapping by relative position
        # Acceptor stem: ~first 10% and last 10%
        # D-loop: ~10-30%
        # Anticodon stem/loop: ~35-55%
        # Variable loop: ~55-65%
        # T-loop: ~65-85%
        if relative_pos < 0.10 or relative_pos > 0.90:
            return tRNARegion.ACCEPTOR_STEM
        elif relative_pos < 0.30:
            return tRNARegion.D_LOOP
        elif relative_pos < 0.55:
            return tRNARegion.ANTICODON_LOOP
        elif relative_pos < 0.65:
            return tRNARegion.VARIABLE_LOOP
        else:
            return tRNARegion.T_LOOP

    def _is_wobble_position(self, position: int, gene: str) -> bool:
        """
        Flag known wobble positions (anticodon position 34).
        These are pre-catalogued for the major disease-associated genes.
        """
        wobble_positions = {
            "MT-TL1": 3254,
            "MT-TK":  8320,
            "MT-TS2": 12236,
        }
        return wobble_positions.get(gene) == position

    def _is_anticodon_position(self, position: int, gene: str) -> bool:
        """Flag positions 34/35/36 of the anticodon loop."""
        anticodon_ranges = {
            "MT-TL1": (3254, 3256),
            "MT-TK":  (8320, 8322),
        }
        if gene in anticodon_ranges:
            start, end = anticodon_ranges[gene]
            return start <= position <= end
        return False
