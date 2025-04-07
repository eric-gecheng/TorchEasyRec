# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class SemanticCross(RankModel):
    """SemanticCross model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self.init_input()

        self.query_embeddings_cnt = self._model_config.query_embeddings_cnt
        self.item_embeddings_cnt = self._model_config.item_embeddings_cnt

        context_feature_dim = self.embedding_group.group_total_dim("context")
        query_feature_dim = self.embedding_group.group_total_dim("query_embeddings")
        item_feature_dim = self.embedding_group.group_total_dim("item_embeddings")
        self.deep_mlp = MLP(
            in_features=context_feature_dim
            + query_feature_dim
            + item_feature_dim
            + self.query_embeddings_cnt * self.item_embeddings_cnt,
            **config_to_kwargs(self._model_config.mlp),
        )
        final_dim = self.deep_mlp.output_dim()

        self.output_mlp = nn.Linear(final_dim, self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)

        context_features = grouped_features["context"]
        query_embeddings = grouped_features["query_embeddings"]
        item_embeddings = grouped_features["item_embeddings"]

        # cross_feature = torch.zeros_like(
        #     query_embeddings
        # )[:, self.query_embeddings_cnt * self.item_embeddings_cnt]

        # cnt = 0
        crosses = []
        for i in range(self.query_embeddings_cnt):
            for j in range(self.item_embeddings_cnt):
                q_emb = query_embeddings[:, i * 1024 : (i + 1) * 1024]
                i_emmb = item_embeddings[:, j * 1024 : (j + 1) * 1024]
                cross = torch.sum(q_emb * i_emmb, dim=1, keepdim=True)
                # cross_feature[:, cnt:cnt+1] = cross
                # cnt += 1
                crosses.append(cross)
        cross_feature = torch.cat(crosses, dim=1)

        y_cat = torch.cat(
            [context_features, query_embeddings, item_embeddings, cross_feature], dim=1
        )
        y_final = self.deep_mlp(y_cat)
        y = self.output_mlp(y_final)

        return self._output_to_prediction(y)
