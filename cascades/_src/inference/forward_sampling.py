# Copyright 2024 The cascades Authors.
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

"""Simple forward sampling from a model."""

from typing import Any
from cascades._src import handlers
from cascades._src import interpreter
from cascades._src.distributions import base as dists
from cascades._src.inference import base
import jax


class ForwardSamplingHook(interpreter.InferenceHook):
  """Inference hook to support simple forward sampling."""

  def handle_observe(self, effect: Any, rng: Any, stats: interpreter.Stats):
    del rng, stats
    # Silently return the event. This will ensure that the observed value
    # is clamped to what it is supposed to be.
    # TODO(charlessutton): Might need to handle rescore observed.
    return effect

  def handle_sample(self, effect: Any, rng: Any, stats: interpreter.Stats):
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


class ForwardSampling(base.Inferencer):
  """Draw samples via forward sampling.

  This class does not allow or handle observations.
  """

  def sample(self, num_samples=1, seed=0):
    """Generate a sample via forward sampling."""
    rng = dists.get_rng(seed)
    samples = interpreter.Samples()
    self._reset_debug_info()
    for sample_idx in range(num_samples):
      rng, subrng = jax.random.split(rng)
      model = self._model(*self._args, **self._kwargs)
      tracer = interpreter.Interpreter(
          model, seed=subrng, inference_hook=ForwardSamplingHook())
      tracer.run()
      self.record_sample(samples, tracer, sample_idx=sample_idx)
    return samples
