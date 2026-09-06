"""Mutable episode-owned prepared features, not public observation content."""
from dataclasses import dataclass
from typing import Any
from copy import deepcopy


@dataclass
class PreparedContext:
    demos: list
    owner: object
    embeddings: Any = None
    positions: Any = None
    source_id: str = 'unrecorded'

    def __setattr__(self,name,value):
        if name in ('demos','owner','source_id') and name in self.__dict__:
            raise AttributeError(f'{name} is immutable; create a new context')
        object.__setattr__(self,name,value)

    def branch_copy(self):
        """Share immutable demo content; copy only mutable feature buffers, never model scratch."""
        def clone(value):
            if value is None:return None
            return value.clone() if hasattr(value,'clone') else deepcopy(value)
        return PreparedContext(self.demos,self.owner,clone(self.embeddings),clone(self.positions),self.source_id)
