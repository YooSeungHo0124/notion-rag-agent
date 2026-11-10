"""Dense (Qwen3-Embedding via sentence-transformers) and sparse (bge-m3 lexical) embedders."""

from __future__ import annotations

from typing import Any, Protocol

from qdrant_client import models

MAX_TOKENS = 2048  # chunks are <=1200 chars; longer text is truncated, not OOM

class DenseEncoder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SparseEncoder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]: ...
    def embed_query(self, text: str) -> models.SparseVector: ...


def _torch_dtype(device: str) -> Any:
    import torch

    return torch.float16 if device.startswith("cuda") else torch.float32


class DenseEmbedder:
    """Qwen3-Embedding wrapper. Queries use the model's ``query`` prompt; documents none."""

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 16) -> None:
        self.model_name, self.device, self.batch_size = model_name, device, batch_size
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype(self.device)}
            if self.device.startswith("cuda"):
                kwargs["device_map"] = self.device  # load weights straight onto the GPU
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=kwargs,
                tokenizer_kwargs={"padding_side": "left"},
            )
            # Qwen3 defaults to 32k tokens; one oversized block would blow up attention memory.
            self._model.max_seq_length = MAX_TOKENS
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        prompts = getattr(self.model, "prompts", {}) or {}
        vec = self.model.encode(
            [text],
            prompt_name="query" if "query" in prompts else None,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].astype("float32").tolist()


def lexical_to_sparse(weights: dict[Any, float]) -> models.SparseVector:
    """Map bge-m3 ``{token_id: weight}`` to a Qdrant SparseVector (sorted, positive only)."""
    items = sorted((int(k), float(v)) for k, v in weights.items() if float(v) > 0)
    return models.SparseVector(indices=[k for k, _ in items], values=[v for _, v in items])


class SparseEmbedder:
    """bge-m3 sparse (lexical weights) encoder via FlagEmbedding."""

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 16) -> None:
        self.model_name, self.device, self.batch_size = model_name, device, batch_size
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_name, use_fp16=self.device.startswith("cuda"), devices=self.device
            )
        return self._model

    def _encode(self, texts: list[str]) -> list[models.SparseVector]:
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=MAX_TOKENS,
        )
        return [lexical_to_sparse(w) for w in out["lexical_weights"]]

    def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return self._encode(texts) if texts else []

    def embed_query(self, text: str) -> models.SparseVector:
        return self._encode([text])[0]
