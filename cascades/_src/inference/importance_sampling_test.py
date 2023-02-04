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

"""Importance sampling test."""

from absl.testing import absltest
from cascades._src import handlers as h
from cascades._src.distributions import base
from cascades._src.inference import importance_sampling
import jax.numpy as jnp
import numpyro.distributions as dists


def _score_fn_max(sofar, value):
  """Use the value as its own score."""
  del sofar
  return value


def _random_seq(vocab_size=10, length=3, score_fn=_score_fn_max):
  """Generate random sequences. Use `score_fn` as likelihood for each choice."""
  seq = []
  for i in range(length):
    choice = yield h.sample(
        dists.Categorical(logits=jnp.zeros(vocab_size)), name=f'choice/{i}'
    )
    if score_fn is not None:
      value_score = score_fn(sofar=seq, value=choice)
      yield h.sample(dist=base.Factor(factor=value_score), obs=choice,
                     name=f'score/{i}')
    seq.append(int(choice))
  return seq


class TracerTest(absltest.TestCase):
  """Test that the beam search tracer runs at all."""

  def test_tracer(self):
    weighted_traces = importance_sampling.sequential_importance_sampling(
        _random_seq, seed=0, nsamples=5, max_rounds=10
    )
    self.assertIsInstance(weighted_traces, list)


if __name__ == '__main__':
  absltest.main()
