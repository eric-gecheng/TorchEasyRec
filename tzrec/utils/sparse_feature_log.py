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


from typing import Dict, List, Set

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def sparse_feature_collect(
    sparse_feature: KeyedJaggedTensor, collector: Dict[str, Set[int]]
):
    """Collect sparse feature ids."""
    names: List[str] = sparse_feature._keys
    for name in names:
        values = sparse_feature[name]._values.cpu().numpy().tolist()
        collector[name].update(values)
