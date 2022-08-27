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

from cascades._src import handlers
from cascades._src.distributions import base as dists
from cascades._src.inference import base
import jax


class RejectionSampling(base.SampledModel):
  """Draw samples via rejection sampling."""

  def __init__(self,
               model,
               *args,
               max_attempts: int = 20,
               observe: Optional[Dict[str, Any]] = None,
               **kwargs):
    """Instantiate a rejection sampler for given model.

    Args:
      model: Model to run sampling on.
      *args: Arguments passed to the model.
      max_attempts: Maximum samples to attempt.
      observe: Values to condition the model on.
      **kwargs: Keyword args passed to the model.

    Returns:
      A sample from the model.
    """
    super().__init__(model=model, *args, observe=observe, **kwargs)
    self._max_attempts = max_attempts

  def sample(self, seed=0):
    """Generate a sample via rejection sampling."""
    rng = dists.get_rng(seed)
    for i in range(self._max_attempts):
      rng, subrng = jax.random.split(rng)
      trace: handlers.Record = self._model(  # pytype: disable=wrong-keyword-args
          *self._args, observe=self._observe,
          **self._kwargs).sample(seed=subrng)
      if not math.isinf(trace.observed_likelihood):
        break
    if math.isinf(trace.observed_likelihood):
      raise ValueError(
          f'Unable to sample successful trace in {self._max_attempts} attempts')
    trace.inputs = (self._args, self._kwargs)
    trace.metadata = dict(attempts=i + 1)
    return trace
