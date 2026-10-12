"""
Variant store — SQLite backend for development/research scale.

Designed to be swapped out for PostgreSQL or a columnar store
(DuckDB, BigQuery) when cohort size demands it. The VariantStore
interface stays the same; only the backend changes.

SQLite is appropriate for:
  - Up to ~5M variants
  - Single-researcher use
  - No concurrent writes

Switch to DuckDB (drop-in, just change connection) for:
  - Analytical queries on 10M+ variants
  - Faster aggregations and feature extraction

Switch to PostgreSQL for:
  - Multi-user / shared lab database
  - Production deployment
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

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

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS variants (
    -- Identity
    variant_id          TEXT PRIMARY KEY,
    source              TEXT NOT NULL,
    variant_type        TEXT NOT NULL,

    -- Genomic coordinates
    chromosome          TEXT,
    position            INTEGER,
    reference_allele    TEXT,
    alternate_allele    TEXT,
    strand              TEXT,

    -- Classification
    pathogenicity       TEXT,
    disease_associations TEXT,   -- JSON array

    -- tRNA annotations (flattened for query performance)
    trna_gene           TEXT,
    isotype             TEXT,
    anticodon           TEXT,
    trna_region         TEXT,
    position_in_trna    INTEGER,
    is_wobble_position  INTEGER, -- 0/1
    is_anticodon_position INTEGER,
    modification_site   TEXT,
    conservation_score  REAL,
    structural_impact   TEXT,
    trf_type            TEXT,
    trf_parent_gene     TEXT,

    -- Variant annotations
    cadd_score          REAL,
    phylop_score        REAL,
    phastcons_score     REAL,
    sift_score          REAL,
    clinvar_id          TEXT,
    clinvar_significance TEXT,
    mitomap_status      TEXT,
    gnomad_af           REAL,
    gnomad_af_popmax    REAL,
    dbsnp_id            TEXT,

    -- Genotype data
    sample_id           TEXT,
    genotype            TEXT,
    allele_depth        TEXT,
    genotype_quality    INTEGER,
    read_depth          INTEGER,
    heteroplasmy_level  REAL,
    is_heteroplasmic    INTEGER,
    haplogroup          TEXT,

    -- Provenance
    raw_record          TEXT,    -- JSON blob of original record
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chromosome_position
    ON variants (chromosome, position);

CREATE INDEX IF NOT EXISTS idx_variant_type
    ON variants (variant_type);

CREATE INDEX IF NOT EXISTS idx_trna_gene
    ON variants (trna_gene);

CREATE INDEX IF NOT EXISTS idx_pathogenicity
    ON variants (pathogenicity);

CREATE INDEX IF NOT EXISTS idx_sample_id
    ON variants (sample_id);

CREATE INDEX IF NOT EXISTS idx_cadd_score
    ON variants (cadd_score);
"""


class VariantStore:
    """
    Persistent store for UnifiedVariant objects.

    Usage:
        store = VariantStore("data/db/variants.db")
        store.insert(variant)
        store.insert_batch(list_of_variants)

        # Query
        mt_variants = store.query(variant_type="mt_trna", min_cadd=20)
        for v in mt_variants:
            print(v)
    """

    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info(f"VariantStore initialized at {self.db_path}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # Better concurrent read performance
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #

    def insert(self, variant: UnifiedVariant) -> bool:
        """Insert a single variant. Returns True if inserted, False if duplicate."""
        row = self._to_row(variant)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO variants VALUES "
                    f"({','.join(['?'] * len(row))})",
                    list(row.values())
                )
            return True
        except sqlite3.Error as e:
            logger.error(f"Store: insert failed for {variant.variant_id}: {e}")
            return False

    def insert_batch(
        self, variants: list[UnifiedVariant], batch_size: int = 1000
    ) -> tuple[int, int]:
        """
        Bulk insert for pipeline use. Returns (inserted, skipped) counts.
        Much faster than individual inserts for large datasets.
        """
        inserted = 0
        skipped = 0

        rows = [self._to_row(v) for v in variants]
        keys = list(rows[0].keys()) if rows else []
        placeholders = ",".join(["?"] * len(keys))

        with self._connect() as conn:
            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                try:
                    cursor = conn.executemany(
                        f"INSERT OR IGNORE INTO variants VALUES ({placeholders})",
                        [list(r.values()) for r in chunk],
                    )
                    inserted += cursor.rowcount
                    skipped += len(chunk) - cursor.rowcount
                except sqlite3.Error as e:
                    logger.error(f"Store: batch insert error: {e}")
                    skipped += len(chunk)

        logger.info(f"Store: inserted {inserted}, skipped {skipped} (duplicates)")
        return inserted, skipped

    def ingest_from_parser(
        self, parser_iterator: Iterator[UnifiedVariant], batch_size: int = 500
    ) -> dict:
        """
        Convenience method: stream variants from a parser directly into the store.

        Usage:
            parser = MitoMapParser()
            stats = store.ingest_from_parser(parser.parse_safe("mitomap.csv"))
        """
        buffer = []
        total_inserted = 0
        total_skipped = 0

        for variant in parser_iterator:
            buffer.append(variant)
            if len(buffer) >= batch_size:
                ins, skip = self.insert_batch(buffer)
                total_inserted += ins
                total_skipped += skip
                buffer = []

        if buffer:
            ins, skip = self.insert_batch(buffer)
            total_inserted += ins
            total_skipped += skip

        return {"inserted": total_inserted, "skipped": total_skipped}

    # ------------------------------------------------------------------ #
    # Read operations
    # ------------------------------------------------------------------ #

    def get(self, variant_id: str) -> Optional[UnifiedVariant]:
        """Fetch a single variant by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM variants WHERE variant_id = ?", (variant_id,)
            ).fetchone()
        return self._from_row(dict(row)) if row else None

    def query(
        self,
        variant_type: Optional[str] = None,
        chromosome: Optional[str] = None,
        trna_gene: Optional[str] = None,
        pathogenicity: Optional[str] = None,
        min_cadd: Optional[float] = None,
        max_gnomad_af: Optional[float] = None,
        sample_id: Optional[str] = None,
        is_wobble_position: Optional[bool] = None,
        trf_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[UnifiedVariant]:
        """
        Flexible query interface. All parameters are optional filters.

        Example:
            # High-confidence pathogenic mt-tRNA anticodon variants
            variants = store.query(
                variant_type="mt_trna",
                pathogenicity="pathogenic",
                min_cadd=20,
                max_gnomad_af=0.001
            )
        """
        conditions = []
        params = []

        if variant_type:
            conditions.append("variant_type = ?")
            params.append(variant_type)
        if chromosome:
            conditions.append("chromosome = ?")
            params.append(chromosome)
        if trna_gene:
            conditions.append("trna_gene = ?")
            params.append(trna_gene)
        if pathogenicity:
            conditions.append("pathogenicity = ?")
            params.append(pathogenicity)
        if min_cadd is not None:
            conditions.append("cadd_score >= ?")
            params.append(min_cadd)
        if max_gnomad_af is not None:
            conditions.append("(gnomad_af IS NULL OR gnomad_af <= ?)")
            params.append(max_gnomad_af)
        if sample_id:
            conditions.append("sample_id = ?")
            params.append(sample_id)
        if is_wobble_position is not None:
            conditions.append("is_wobble_position = ?")
            params.append(1 if is_wobble_position else 0)
        if trf_type:
            conditions.append("trf_type = ?")
            params.append(trf_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""
        sql = f"SELECT * FROM variants {where} ORDER BY chromosome, position {limit_clause}"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._from_row(dict(r)) for r in rows]

    def stats(self) -> dict:
        """Summary statistics of the current store contents."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM variants").fetchone()[0]
            by_type = dict(
                conn.execute(
                    "SELECT variant_type, COUNT(*) FROM variants GROUP BY variant_type"
                ).fetchall()
            )
            by_pathogenicity = dict(
                conn.execute(
                    "SELECT pathogenicity, COUNT(*) FROM variants GROUP BY pathogenicity"
                ).fetchall()
            )
            by_gene = conn.execute(
                "SELECT trna_gene, COUNT(*) as n FROM variants "
                "WHERE trna_gene IS NOT NULL "
                "GROUP BY trna_gene ORDER BY n DESC LIMIT 10"
            ).fetchall()

        return {
            "total_variants": total,
            "by_type": by_type,
            "by_pathogenicity": by_pathogenicity,
            "top_genes": [dict(r) for r in by_gene],
        }

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    def _to_row(self, v: UnifiedVariant) -> dict:
        """Flatten a UnifiedVariant into a database row dict."""
        ta = v.trna_annotation or tRNAAnnotation()
        va = v.variant_annotation or VariantAnnotation()
        gd = v.genotype_data or GenotypeData()

        return {
            "variant_id":           v.variant_id,
            "source":               v.source,
            "variant_type":         v.variant_type.value,
            "chromosome":           v.chromosome,
            "position":             v.position,
            "reference_allele":     v.reference_allele,
            "alternate_allele":     v.alternate_allele,
            "strand":               v.strand.value,
            "pathogenicity":        v.pathogenicity.value,
            "disease_associations": json.dumps(v.disease_associations),
            "trna_gene":            ta.trna_gene,
            "isotype":              ta.isotype,
            "anticodon":            ta.anticodon,
            "trna_region":          ta.trna_region.value if ta.trna_region else None,
            "position_in_trna":     ta.position_in_trna,
            "is_wobble_position":   int(ta.is_wobble_position),
            "is_anticodon_position":int(ta.is_anticodon_position),
            "modification_site":    ta.modification_site,
            "conservation_score":   ta.conservation_score,
            "structural_impact":    ta.structural_impact,
            "trf_type":             ta.trf_type,
            "trf_parent_gene":      ta.trf_parent_gene,
            "cadd_score":           va.cadd_score,
            "phylop_score":         va.phylop_score,
            "phastcons_score":      va.phastcons_score,
            "sift_score":           va.sift_score,
            "clinvar_id":           va.clinvar_id,
            "clinvar_significance": va.clinvar_significance,
            "mitomap_status":       va.mitomap_status,
            "gnomad_af":            va.gnomad_af,
            "gnomad_af_popmax":     va.gnomad_af_popmax,
            "dbsnp_id":             va.dbsnp_id,
            "sample_id":            gd.sample_id,
            "genotype":             gd.genotype,
            "allele_depth":         gd.allele_depth,
            "genotype_quality":     gd.genotype_quality,
            "read_depth":           gd.read_depth,
            "heteroplasmy_level":   gd.heteroplasmy_level,
            "is_heteroplasmic":     int(gd.is_heteroplasmic) if gd.is_heteroplasmic is not None else None,
            "haplogroup":           gd.haplogroup,
            "raw_record":           json.dumps(v.raw_record) if v.raw_record else None,
            "created_at":           None,  # handled by DB default
        }

    def _from_row(self, row: dict) -> UnifiedVariant:
        """Reconstruct a UnifiedVariant from a database row."""
        trna_region_val = row.get("trna_region")
        trna_annot = tRNAAnnotation(
            trna_gene=row.get("trna_gene"),
            isotype=row.get("isotype"),
            anticodon=row.get("anticodon"),
            trna_region=tRNARegion(trna_region_val) if trna_region_val else None,
            position_in_trna=row.get("position_in_trna"),
            is_wobble_position=bool(row.get("is_wobble_position")),
            is_anticodon_position=bool(row.get("is_anticodon_position")),
            modification_site=row.get("modification_site"),
            conservation_score=row.get("conservation_score"),
            structural_impact=row.get("structural_impact"),
            trf_type=row.get("trf_type"),
            trf_parent_gene=row.get("trf_parent_gene"),
        )

        variant_annot = VariantAnnotation(
            cadd_score=row.get("cadd_score"),
            phylop_score=row.get("phylop_score"),
            phastcons_score=row.get("phastcons_score"),
            sift_score=row.get("sift_score"),
            clinvar_id=row.get("clinvar_id"),
            clinvar_significance=row.get("clinvar_significance"),
            mitomap_status=row.get("mitomap_status"),
            gnomad_af=row.get("gnomad_af"),
            gnomad_af_popmax=row.get("gnomad_af_popmax"),
            dbsnp_id=row.get("dbsnp_id"),
        )

        is_het = row.get("is_heteroplasmic")
        genotype_data = GenotypeData(
            sample_id=row.get("sample_id"),
            genotype=row.get("genotype"),
            allele_depth=row.get("allele_depth"),
            genotype_quality=row.get("genotype_quality"),
            read_depth=row.get("read_depth"),
            heteroplasmy_level=row.get("heteroplasmy_level"),
            is_heteroplasmic=bool(is_het) if is_het is not None else None,
            haplogroup=row.get("haplogroup"),
        )

        raw = row.get("raw_record")

        return UnifiedVariant(
            variant_id=row["variant_id"],
            source=row["source"],
            variant_type=VariantType(row["variant_type"]),
            chromosome=row.get("chromosome"),
            position=row.get("position"),
            reference_allele=row.get("reference_allele"),
            alternate_allele=row.get("alternate_allele"),
            strand=Strand(row.get("strand", ".")),
            pathogenicity=PathogenicityClass(row.get("pathogenicity", "unknown")),
            disease_associations=json.loads(row["disease_associations"]) if row.get("disease_associations") else [],
            trna_annotation=trna_annot,
            variant_annotation=variant_annot,
            genotype_data=genotype_data,
            raw_record=json.loads(raw) if raw else None,
        )
