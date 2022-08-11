# Copyright 2022 The cascades Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base types for inference."""

import dataclasses
from typing import Any, Dict, Text, Optional


@dataclasses.dataclass
class InferenceResult:
  metadata: Any = None
  output: Any = None
  trace: Any = None
  inputs: Any = None

  def __getitem__(self, k):
    return self.trace[k]


class InferenceAlgorithm:
  """Base class for inference methods."""

  def __init__(self,
               model,
               *args,
               observe: Optional[Dict[Text, Any]] = None,
               **kwargs):
    """Run inference on model called with given inputs.

    Args:
      model: Model to run sampling on.
      *args: Arguments passed to the model.
      observe: Values to condition the model on.
      **kwargs: Keyword args passed to the model.

    Returns:
      A sample from the model.
    """
    self.model = model
    self.args = args
    self.kwargs = kwargs
    self.observe = observe

  def sample(self):
    raise NotImplementedError
