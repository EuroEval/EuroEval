"""Maintainer-side validation and promotion of staged volunteer results."""

from __future__ import annotations

import os
import typing as t

from huggingface_hub import HfApi

from leaderboards.canonical_publication import BucketApi as CanonicalBucketApi
from leaderboards.canonical_publication import CanonicalPublisher
from leaderboards.constants import HF_RESULTS_BUCKET

from .review_models import (
    BrokerBinder,
    BrokerPromoter,
    BrokerRenewer,
    BrokerReservation,
    BrokerReservationResult,
    BucketApi,
    BucketEntry,
    BucketInfo,
    JsonObject,
    PublicStagingError,
    ReviewError,
    ReviewReport,
    ValidatedRecord,
)
from .review_storage import BucketStore
from .review_transaction import (
    VolunteerReviewer,
    bind_with_broker,
    promote_with_broker,
    renew_with_broker,
    reserve_with_broker,
)
from .review_validation import load_scope_policy

__all__ = [
    "BucketApi",
    "BucketEntry",
    "BucketInfo",
    "BucketStore",
    "BrokerBinder",
    "BrokerPromoter",
    "BrokerRenewer",
    "BrokerReservation",
    "BrokerReservationResult",
    "JsonObject",
    "PublicStagingError",
    "ReviewError",
    "ReviewReport",
    "ValidatedRecord",
    "VolunteerReviewer",
    "load_scope_policy",
    "bind_with_broker",
    "promote_with_broker",
    "renew_with_broker",
    "reserve_with_broker",
    "reviewer_from_environment",
]


def reviewer_from_environment() -> VolunteerReviewer:
    """Build the configured maintainer reviewer service.

    Returns:
        A reviewer connected to the configured staging and canonical buckets.

    Raises:
        ReviewError:
            If the staging bucket or token is not configured.
    """
    staging = os.environ.get("HF_STAGING_BUCKET")
    token = os.environ.get("HF_TOKEN")
    if not staging or not token:
        raise ReviewError("HF_STAGING_BUCKET and HF_TOKEN are required")
    api = HfApi(token=token)
    results_bucket = os.environ.get("HF_RESULTS_BUCKET", HF_RESULTS_BUCKET)
    return VolunteerReviewer(
        store=BucketStore(
            api=t.cast(BucketApi, api), token=token, staging_bucket=staging
        ),
        results_bucket=results_bucket,
        publisher_factory=lambda: CanonicalPublisher(
            api=t.cast(CanonicalBucketApi, api), token=token, bucket=results_bucket
        ),
    )
