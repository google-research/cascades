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

"""Rejection sampling from a model."""

import math
from typing import Any, Dict, Optional

from cascades._src.distributions import base as dists
from cascades._src.inference import base
import jax


class RejectionSampling(base.InferenceAlgorithm):
  """Draw samples via rejection sampling."""

  def __init__(self,
               model,
               *args,
               observe: Optional[Dict[str, Any]] = None,
               max_attempts=20,
               **kwargs):
    """Instantiate a rejection sampler for given model.

    Args:
      model: Model to run sampling on.
      *args: Arguments passed to the model.
      observe: Values to condition the model on.
      max_attempts: Maximum samples to attempt.
      **kwargs: Keyword args passed to the model.

    Returns:
      A sample from the model.
    """
    super().__init__(model=model, *args, observe=observe, **kwargs)
    self.max_attempts = max_attempts

  def sample(self, seed=0):
    """Generate a sample via rejection sampling."""
    output = None
    rng = dists.get_rng(seed)
    for i in range(self.max_attempts):
      rng, subrng = jax.random.split(rng)
      trace = self.model.sample(
          *self.args, seed=subrng, observe=self.observe, **self.kwargs)
      if not math.isinf(trace.observed_likelihood):
        output = trace.return_value
        break
    return base.InferenceResult(
        metadata=dict(attempts=i + 1),
        trace=trace,
        inputs=(self.args, self.kwargs),
        output=output)
