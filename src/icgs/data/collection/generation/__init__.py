"""Canonical ICGS data-generation protocol.

Serialized schema identifiers remain stable for artifact compatibility.
"""

from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL, GenerationProtocol

__all__ = ["GENERATION_PROTOCOL", "GenerationProtocol"]
