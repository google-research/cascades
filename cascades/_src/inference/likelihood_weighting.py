# Copyright 2023 The cascades Authors.
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

"""Likelihood weighting inference algorithm."""

from typing import Any, Dict

from cascades._src import handlers
from cascades._src import interpreter
from cascades._src.distributions import base as dists
from cascades._src.inference import base
import jax


# TODO(charlessutton): Provide easy way to get marginal likelihood.
class LikelihoodWeightingHook(interpreter.InferenceHook):
  """InferenceHook for the interpreter to support likelihood weighting."""

  def __init__(self, observed, await_timeout=60):
    super().__init__()
    self._observed = observed
    self._observed_used: Dict[str, bool] = {}
    self._await_timeout = await_timeout

  # effect is either a Sample or an Observe.
  # TODO(charlessutton): Should we handle logs?
  def handle_observe(self, effect: Any, rng: Any, stats: interpreter.Stats):
    del rng
    observed_value = self._observed[effect.name]
    # TODO(charlessutton): Maybe raise an error if not all observed vars used.
    # (This could be on a `done` callback.)
    self._observed_used[effect.name] = True
    effect.value = observed_value
    effect.score = handlers.dists.score_distribution(
        effect.fn,
        effect.value,
        await_timeout=self._await_timeout)
    stats.likelihood_observed += effect.score
    return effect

  def handle_sample(self, effect: Any, rng: Any, stats: interpreter.Stats):
    # A Sample effect is equivalent to an Observe if the name is in
    # self._observe.
    if effect.name in self._observed:
      return self.handle_observe(effect, rng, stats)

    effect.kwargs['rng'] = rng
    if effect.value is None:
      random_sample = handlers.dists.sample_distribution(
          effect.fn,
          *effect.args,
          await_timeout=self._await_timeout,
          **effect.kwargs)
      effect.value = random_sample.value
      effect.score = random_sample.log_p
    if effect.score is not None:
      stats.likelihood_sampled += effect.score
    return effect


class LikelihoodWeighting(base.Inferencer):
  """Draw samples via likelihood weighting.

  This is importance sampling, where the proposal distribution is the prior
  distribution give by the cascade.

  This will return a weighted Samples.
  """

  def __init__(self, model, *args, observed=None, **kwargs):
    super().__init__(model, *args, **kwargs)
    if observed is None:
      raise ValueError('No variables specified to observe.')
    self._observed = observed

  def sample(self, num_samples=1, seed=0):
    """Generate a sample via likelihood weighting."""
    rng = dists.get_rng(seed)
    samples = interpreter.Samples()
    self._reset_debug_info()
    for _ in range(num_samples):
      rng, subrng = jax.random.split(rng)
      model = self._model(*self._args, **self._kwargs)
      tracer = interpreter.Interpreter(
          model,
          seed=subrng,
          inference_hook=LikelihoodWeightingHook(self._observed))
      tracer.run()
      self.record_sample(samples, tracer)
    return samples
