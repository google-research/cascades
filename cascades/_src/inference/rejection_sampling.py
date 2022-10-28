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

"""Rejection sampling from a cascades model."""

import math
from typing import Any, Dict, Optional

from cascades._src import handlers
from cascades._src import interpreter
from cascades._src.distributions import base as dists
from cascades._src.inference import base
import jax


class RejectionSamplingHook(interpreter.InferenceHook):
  """Inference hook to allow interpreter to do rejection sampling."""

  def __init__(self, observed, await_timeout=60):
    super().__init__()
    self._observed = observed
    self._observed_used = dict()
    self._await_timeout = await_timeout

  # effect is either a Sample or an Observe.
  # TODO(charlessutton): Should we handle logs?
  def handle_observe(self, effect: Any, rng: Any, stats: interpreter.Stats):
    observed_value = self._observed[effect.name]
    # Generate the putative sample.
    effect.kwargs['rng'] = rng
    random_sampler = handlers.dists.sample_distribution(
        effect.fn,
        *effect.args,
        await_timeout=self._await_timeout,
        **effect.kwargs)
    effect.value = random_sampler.value
    if observed_value != random_sampler.value:
      # Reject. Sample did not match the observation.
      effect.score = -jax.numpy.inf
      stats.likelihood_observed = -jax.numpy.inf
    else:
      # We did match, so we are allowed to continue.
      effect.score = random_sampler.log_p
      stats.likelihood_observed += effect.score
    return effect

  def handle_sample(self, effect: Any, rng: Any, stats: interpreter.Stats):
    # A Sample effect is equivalent to an Observe if the name is in
    # self._observe.
    if effect.name in self._observed:
      return self.handle_observe(effect, rng, stats)

    # This wasn't an observe. Just sample the value.
    effect.kwargs['rng'] = rng
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

  def handle_log(self, effect: Any, rng: Any, stats: interpreter.Stats):
    # A Log effect is equivalent to an Observe if the name is in
    # self._observe.
    if effect.name in self._observed:
      return self.handle_observe(effect, rng, stats)
    else:
      return effect


class RejectionSampling(base.Inferencer):
  """Draw samples via rejection sampling."""

  def __init__(self,
               model,
               *args,
               max_attempts: int = 20,
               observed: Optional[Dict[str, Any]] = None,
               **kwargs):
    """Instantiate a rejection sampler for given model.

    Args:
      model: Model to run sampling on.
      *args: Arguments passed to the model.
      max_attempts: Maximum samples to attempt.
      observed: Values to condition the model on.
      **kwargs: Keyword args passed to the model.

    Returns:
      A sample from the model.
    """
    super().__init__(model, *args, **kwargs)
    if observed is None:
      raise ValueError('No variables specified to observe.')
    self._observed = observed
    self._max_attempts = max_attempts

  def sample(self, num_samples=1, seed=0):
    """Generate a sample via rejection sampling."""
    rng = dists.get_rng(seed)
    samples = interpreter.Samples()
    for sample_idx in range(num_samples):
      for attempt_idx in range(self._max_attempts):
        rng, subrng = jax.random.split(rng)
        model = self._model(*self._args, **self._kwargs)
        tracer = interpreter.Interpreter(
            model,
            seed=subrng,
            inference_hook=RejectionSamplingHook(self._observed))
        tracer.run()
        success = not math.isinf(tracer.stats.likelihood_observed)
        if success: break
      if success:
        self.record_sample(samples, tracer)
        samples.update_row(dict(num_attempts=1 + attempt_idx))
      else:
        # Rejection sampling failed. Add N/A values for everything.
        samples.append_row(
            dict(sample_idx=sample_idx, num_attempts=1 + attempt_idx))
    return samples
