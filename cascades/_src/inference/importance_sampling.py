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

"""Importance sampling variants."""
from cascades._src.inference import beam_search
import jax

Tracer = beam_search.Tracer


def _normalize_weights(tracers):
  weights = jax.numpy.array(
      [tracer.handlers.observed_likelihood for tracer in tracers])
  weight_sum = jax.scipy.special.logsumexp(weights)
  normalized_weights = [w - weight_sum for w in weights]
  return normalized_weights


def sequential_importance_sampling(fn, seed, nsamples=5, max_rounds=20):
  """Sequential importance sampling.

  Args:
    fn: Cascade function to run.
    seed: Seed for this run.
    nsamples: Number of beams in active set.
    max_rounds: Maximum length of trajectories. Used to prevent runaway
      generations.

  Returns:
    List of (weight, traces)
  """
  completed_traces = []

  seeds = jax.random.split(jax.random.PRNGKey(seed), nsamples)
  active_traces = [Tracer.new(fn=fn, seed=seed) for seed in seeds]

  for _ in range(max_rounds):
    new_active_traces = []
    for tracer in active_traces:
      tracer.run_until_observe()
      if tracer.done:
        completed_traces.append(tracer)
      else:
        new_active_traces.append(tracer)

    active_traces = new_active_traces
    if not active_traces:
      break

  normalized_weights = _normalize_weights(completed_traces)

  return list(zip(normalized_weights, completed_traces))
