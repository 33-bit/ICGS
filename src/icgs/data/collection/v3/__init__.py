"""ICGS primary v3 collection protocol.

This package is additive.  It must not mutate primary v2 manifests, profiles,
or the v1 episode schema.
"""

from icgs.data.collection.v3.protocol import V3_PROTOCOL, V3Protocol

__all__ = ["V3_PROTOCOL", "V3Protocol"]
