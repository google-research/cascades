# Copyright 2026 The cascades Authors.
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

"""Choose k items from a set."""

import dataclasses
from typing import Any, List, Tuple, Union

from cascades._src.distributions import base
import jax
import jax.numpy as jnp


def _sample_log_probs(rng, log_probs, k=None):
  # Sample categorical distribution of log probs
  noise = jax.random.gumbel(rng, shape=log_probs.shape)
  perturbed = log_probs + noise
  if k is None:
    return jnp.argmax(perturbed, axis=-1)
  else:
    return jnp.argsort(perturbed, axis=-1)[:k]


@dataclasses.dataclass(frozen=True, eq=True)
class Choose(base.Distribution):
  """Choose an ordered sample of k options uniformly without replacement."""
  k: int = 1
  options: Union[Tuple[Any], List[Any]] = tuple()

  def sample(self, rng) -> base.RandomSample:
    rng = base.get_rng(rng)
    n = len(self.options)
    if not 0 <= self.k <= n:
      raise ValueError(f'k must be between 0 and {n}, got {self.k}')
    if self.k == 0:
      return base.RandomSample(value=[], log_p=0.0)

    log_p = jnp.log(1.0 / n)
    log_probs = jnp.full((n,), fill_value=log_p)

    chosen_idxs = _sample_log_probs(rng, log_probs=log_probs, k=self.k)
    chosen = []
    for idx in chosen_idxs:
      chosen.append(self.options[idx])
    # Each draw has one fewer available option than the previous draw.
    sample_log_p = -jnp.log(jnp.arange(n - self.k + 1, n + 1)).sum()
    return_value = base.RandomSample(value=chosen, log_p=sample_log_p)
    return return_value
