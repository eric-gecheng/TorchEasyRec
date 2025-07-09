# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    EmbeddingCollectionInterface,
    get_embedding_names_by_table,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


@torch.fx.wrap
def get_zeros_collection(f: JaggedTensor, dim: int):
    """Get zeros tensor."""
    n_samples = len(f.values())
    return torch.zeros((n_samples, dim), dtype=torch.float, device="cpu")


class MockEmbeddingCollection(EmbeddingCollectionInterface):
    """mock embedding collection."""

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._embedding_configs = tables
        self._device: torch.device = (
            device if device is not None else torch.device("cpu")
        )

        self._embedding_names_by_table: List[List[str]] = get_embedding_names_by_table(
            tables
        )
        self._feature_names: List[List[str]] = [table.feature_names for table in tables]

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        """Forward pass."""
        feature_embeddings: Dict[str, JaggedTensor] = {}
        jt_dict: Dict[str, JaggedTensor] = features.to_dict()
        for i, emb_config in enumerate(self._embedding_configs):
            feature_names = self._feature_names[i]
            embedding_names = self._embedding_names_by_table[i]
            for j, embedding_name in enumerate(embedding_names):
                feature_name = feature_names[j]
                f = jt_dict[feature_name]
                dim = emb_config.embedding_dim
                lookup = get_zeros_collection(f, dim)
                feature_embeddings[embedding_name] = JaggedTensor(
                    values=lookup, lengths=f.lengths()
                )
        return feature_embeddings

    def need_indices(self) -> bool:
        """Override."""
        pass

    def embedding_dim(self) -> int:
        """Override."""
        pass

    def embedding_configs(self) -> List[EmbeddingConfig]:
        """Returns: List[EmbeddingConfig]: The embedding configs."""
        return self._embedding_configs

    def embedding_names_by_table(self) -> List[List[str]]:
        """Returns: List[List[str]]: The embedding names by table."""
        return self._embedding_names_by_table

    @property
    def device(self) -> torch.device:
        """Returns: torch.device: The compute device."""
        return self._device


@torch.fx.wrap
def get_zeros_bag(f: JaggedTensor, dim: int):
    """Get zeros tensor."""
    n_samples = len(f.offsets())
    return torch.zeros((n_samples, dim), dtype=torch.float, device="cpu")


class MockEmbeddingBagCollection(EmbeddingBagCollectionInterface):
    """Mock embedding bag collection."""

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._is_weighted = is_weighted
        self._embedding_bag_configs = tables
        self._device: torch.device = device or torch.device("cpu")
        self._lengths_per_embedding: List[int] = []

        self._feature_names: List[List[str]] = [table.feature_names for table in tables]
        self._embedding_names: List[str] = [
            embedding
            for embeddings in get_embedding_names_by_table(tables)
            for embedding in embeddings
        ]

        for embedding_config in tables:
            if not embedding_config.feature_names:
                embedding_config.feature_names = [embedding_config.name]
            self._lengths_per_embedding.extend(
                len(embedding_config.feature_names) * [embedding_config.embedding_dim]
            )

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        """Forward pass.

        Args:
            features (KeyedJaggedTensor): Input KJT
        Returns:
            KeyedTensor
        """
        pooled_emb = []
        features_dict = features.to_dict()
        for i, _ in enumerate(self._embedding_bag_configs):
            dim = self._lengths_per_embedding[i]
            name = self._feature_names[i][0]
            f = features_dict[name]
            pooled_emb.append(get_zeros_bag(f, dim))

        pooled_emb = torch.cat(pooled_emb, dim=1)

        return KeyedTensor(
            keys=self._embedding_names,
            values=pooled_emb,
            length_per_key=self._lengths_per_embedding,
        )

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        """Returns List[EmbeddingBagConfig]: The embedding bag configs."""
        return self._embedding_bag_configs

    @property
    def device(self) -> torch.device:
        """Returns: torch.device: The compute device."""
        return self._device

    def is_weighted(self) -> bool:
        """Returns:bool: Whether the EmbeddingBagCollection is weighted."""
        return self._is_weighted
