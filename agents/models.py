from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PinData(BaseModel):
    """A single Pinterest pin with all extracted metadata."""

    pin_id: str = Field(..., description="Unique Pinterest pin identifier")
    title: Optional[str] = Field(None, description="Pin title")
    description: Optional[str] = Field(None, description="Pin description")
    image_url: str = Field(..., description="Primary image URL")
    original_image_url: Optional[str] = Field(None, description="High-res image URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail image URL")
    board_name: Optional[str] = Field(None, description="Board name")
    creator_name: Optional[str] = Field(None, description="Creator username")
    creator_url: Optional[str] = Field(None, description="Creator profile URL")
    pin_url: str = Field(..., description="Pinterest pin page URL")
    external_link: Optional[str] = Field(None, description="External link on the pin")
    domain: Optional[str] = Field(None, description="Domain of external link")
    alt_text: Optional[str] = Field(None, description="Image alt text")
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when scraped",
    )
    raw_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw extracted data for debugging",
    )

    # Phase 2 — Image Analysis Agent populates this
    image_analysis: Optional["ImageAnalysisResult"] = None
    # dominant_colors: Optional[List[str]] = None
    # image_embedding: Optional[List[float]] = None

    # Phase 3 — Ranking Agent populates these
    rank_score: Optional[float] = None
    rank_position: Optional[int] = None
    rank_breakdown: Optional[Dict[str, float]] = None

    # Phase 4 — Quality Evaluation Agent populates this
    passed_quality: Optional[bool] = None

    # Phase 5 — Instagram Publisher Agent populates this
    instagram_post: Optional["InstagramPost"] = None

class ImageAnalysisResult(BaseModel):
    """Analysis of a single pin's image — local + LLM."""

    pin_id: str = Field(..., description="Links the analysis to the pin")

    # ── Local analysis (no API) ────────────────────────────────────────
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    aspect_ratio: Optional[float] = Field(None, description="Width / height")
    dominant_colors: List[str] = Field(
        default_factory=list, description="Top colors in hex format"
    )
    orientation: Optional[str] = Field(
        None, description="portrait / landscape / square"
    )

    # ── LLM analysis (Claude vision) ────────────────────────────────────
    description: Optional[str] = Field(None, description="Image description from the LLM")
    tags: List[str] = Field(default_factory=list, description="Content tags")
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Quality score 0-1 from the LLM"
    )

    # ── Metadata ────────────────────────────────────────────────────────
    analyzed_by_llm: bool = Field(False, description="Whether LLM analysis succeeded")
    error: Optional[str] = Field(None, description="Analysis error, if any")


class PinterestScrapingResult(BaseModel):
    """Result of a complete Pinterest scraping operation."""

    query: str = Field(..., description="Search query used")
    pins: List[PinData] = Field(default_factory=list, description="Scraped pins")
    total_scraped: int = Field(0, description="Number of successfully scraped pins")
    total_attempted: int = Field(0, description="Target pin count")
    scraping_duration_seconds: float = Field(0.0, description="Duration in seconds")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the scraping operation",
    )

    @property
    def success_rate(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.total_scraped / self.total_attempted

    def summary(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_scraped": self.total_scraped,
            "total_attempted": self.total_attempted,
            "success_rate": f"{self.success_rate:.1%}",
            "duration_seconds": round(self.scraping_duration_seconds, 2),
            "error_count": len(self.errors),
        }

class RankingResult(BaseModel):
    """Phase 3 — summary of the ranking pass."""

    query: str = Field(..., description="Query these pins belong to")
    total_ranked: int = Field(0, description="How many pins were scored")
    ranked_pin_ids: List[str] = Field(
        default_factory=list, description="Pin IDs ordered best-first"
    )
    top_pin_id: Optional[str] = Field(None, description="Highest scoring pin")
    weights_used: Dict[str, float] = Field(
        default_factory=dict, description="Scoring weights applied"
    )

    def summary(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_ranked": self.total_ranked,
            "top_pin_id": self.top_pin_id,
            "weights_used": self.weights_used,
        }


class QualityEvaluationResult(BaseModel):
    """Phase 4 — batch-level quality report + publish gate."""

    query: str = Field(..., description="Query these pins belong to")
    threshold: float = Field(..., description="Minimum rank_score to pass")
    total_evaluated: int = Field(0, description="Pins evaluated")
    passed: int = Field(0, description="Pins at/above threshold")
    rejected: int = Field(0, description="Pins below threshold")
    average_score: Optional[float] = Field(None, description="Mean rank_score")
    passed_pin_ids: List[str] = Field(default_factory=list, description="Best-first")
    batch_ready: bool = Field(False, description="Enough quality pins to publish")

    def summary(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "threshold": self.threshold,
            "total_evaluated": self.total_evaluated,
            "passed": self.passed,
            "rejected": self.rejected,
            "average_score": self.average_score,
            "batch_ready": self.batch_ready,
        }


class InstagramPost(BaseModel):
    """Phase 5 — a publish-ready post prepared for one pin."""

    pin_id: str = Field(..., description="Source pin")
    image_url: str = Field(..., description="Image to publish")
    caption: str = Field(..., description="Full caption text")
    hashtags: List[str] = Field(default_factory=list, description="Hashtags used")
    status: str = Field("prepared", description="prepared / published / failed / skipped")
    would_publish: bool = Field(True, description="True in dry-run (not actually posted)")
    published_id: Optional[str] = Field(None, description="Real post id when published")
    error: Optional[str] = Field(None, description="Publish error if any")


class InstagramPublishResult(BaseModel):
    """Phase 5 — summary of the publishing pass."""

    query: str = Field(..., description="Query these posts belong to")
    dry_run: bool = Field(True, description="True if nothing was actually posted")
    total_prepared: int = Field(0, description="Posts prepared")
    total_published: int = Field(0, description="Posts actually published")
    posts: List[InstagramPost] = Field(default_factory=list, description="Prepared posts")

    def summary(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "dry_run": self.dry_run,
            "total_prepared": self.total_prepared,
            "total_published": self.total_published,
        }
