# -*- coding: utf-8 -*-
"""Chunker base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rag_chunk.types_ import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for chunkers."""

    name: str = "base"

    @abstractmethod
    def chunk(self, doc: Document) -> List[Chunk]:
        """Split a document into chunks."""
        raise NotImplementedError