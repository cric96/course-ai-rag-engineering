"""Metadata filters: search only among the chunks we already know are relevant."""

from qdrant_client import models


def metadata_filter(**fields) -> models.Filter:
    """Qdrant filter on Document metadata, e.g. metadata_filter(call="bando-x")."""
    return models.Filter(
        must=[
            # LangChain stores each Document's metadata under the "metadata" key.
            models.FieldCondition(
                key=f"metadata.{key}", match=models.MatchValue(value=value)
            )
            for key, value in fields.items()
        ]
    )
