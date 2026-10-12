"""
Abstract base class for all data source parsers.

Every new data source (MitoMap, gnomAD, UK Biobank VCF, etc.)
implements this interface. The rest of the pipeline only ever
sees UnifiedVariant objects — it doesn't care where they came from.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from src.schema.variant import UnifiedVariant

logger = logging.getLogger(__name__)


class ParserError(Exception):
    """Raised when a parser encounters unrecoverable data issues."""
    pass


class BaseParser(ABC):
    """
    All parsers inherit from this. Implement parse() and that's it.

    Design principle: parsers are generators. They yield one UnifiedVariant
    at a time rather than loading everything into memory. This keeps the
    harness usable on large WGS cohorts without RAM issues.
    """

    def __init__(self, source_name: str, config: Optional[dict] = None):
        self.source_name = source_name
        self.config = config or {}
        self._parsed_count = 0
        self._skipped_count = 0
        self._error_count = 0

    @abstractmethod
    def parse(self, source: Path | str) -> Iterator[UnifiedVariant]:
        """
        Parse a data source and yield UnifiedVariant objects.

        Args:
            source: File path, URL, or database connection string.

        Yields:
            UnifiedVariant objects, one per variant record.
        """
        pass

    @abstractmethod
    def validate_source(self, source: Path | str) -> bool:
        """
        Check that a source file/URL is accessible and in the expected format
        before committing to a full parse. Should be fast.
        """
        pass

    def parse_safe(self, source: Path | str) -> Iterator[UnifiedVariant]:
        """
        Wrapper around parse() that catches per-record errors and logs them
        rather than crashing the whole pipeline. Use this in production.
        """
        if not self.validate_source(source):
            raise ParserError(f"{self.source_name}: source validation failed for {source}")

        for record in self.parse(source):
            try:
                yield record
                self._parsed_count += 1
            except Exception as e:
                self._error_count += 1
                logger.warning(f"{self.source_name}: skipped record due to error: {e}")

    def stats(self) -> dict:
        return {
            "source": self.source_name,
            "parsed": self._parsed_count,
            "skipped": self._skipped_count,
            "errors": self._error_count,
        }

    def _make_variant_id(self, chrom: str, pos: int, ref: str, alt: str) -> str:
        """Standard variant ID format: source::chrom-pos-ref-alt"""
        return f"{self.source_name}::{chrom}-{pos}-{ref}-{alt}"
