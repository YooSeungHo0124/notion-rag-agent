"""Cross-encoder reranker (bge-reranker-v2-m3) with sigmoid-normalised scores in 0..1."""

from __future__ import annotations

from typing import Any, Protocol


class RerankerProtocol(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class Reranker:
    """FlagEmbedding FlagReranker wrapper; lazy-loads the model."""

    def __init__(
        self, model_name: str, device: str = "cuda", batch_size: int = 16, max_length: int = 1024
    ) -> None:
        self.model_name, self.device = model_name, device
        self.batch_size, self.max_length = batch_size, max_length
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self.model_name, use_fp16=self.device.startswith("cuda"), devices=self.device
            )
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.compute_score(
            [[query, p] for p in passages],
            normalize=True,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        if isinstance(scores, float | int):
            scores = [scores]
        return [float(s) for s in scores]
