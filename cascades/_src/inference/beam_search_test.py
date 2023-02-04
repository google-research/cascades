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

"""Tests for beam_search."""

from absl.testing import absltest
from cascades._src import handlers as h
from cascades._src.distributions import base
from cascades._src.inference import beam_search
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
    t = beam_search.Tracer.new(fn=_random_seq, seed=0)
    t.reify()
    self.assertIsInstance(t.result, list)


class BeamSearchTest(absltest.TestCase):
  """Test that beam search runs."""

  def _run_beam_search(self,
                       score_fn,
                       nbeams=3,
                       length=4,
                       factor=5,
                       intermediates=True):
    result = beam_search.beam_search_by_score(
        lambda: h.AutoName(_random_seq(length=length))(),  # pylint: disable=unnecessary-lambda
        seed=0,
        score_fn=score_fn,
        nbeams=nbeams,
        expansion_factor=factor,
        return_intermediates=intermediates,
    )
    # At each observe + (1 for the run until end),
    # each of nbeams is cloned (factor - 1) times.
    # Add in the original nbeams
    expected = nbeams * (factor - 1) * (length + 1) + nbeams

    if intermediates:
      self.assertLen(result['intermediates'], expected)
      completed = result['completed']
    else:
      completed = result
    self.assertLen(completed, nbeams)
    return result

  def test_beam_search(self, nbeams=3, length=4, factor=5):
    self._run_beam_search(
        score_fn=beam_search.score_fn_last,
        nbeams=nbeams,
        length=length,
        factor=factor)

  def test_beam_search_all_scores(self, nbeams=3, length=4, factor=5):
    self._run_beam_search(
        score_fn=beam_search.score_fn_all,
        nbeams=nbeams,
        length=length,
        factor=factor)

#  def test_score_fns(self, nbeams=1, length=5, factor=5):
#    all_beams = self._run_beam_search(
#        score_fn=beam_search.score_fn_all,
#        nbeams=nbeams,
#        length=length,
#        factor=factor,
#        intermediates=False)
#    last_beams = self._run_beam_search(
#        score_fn=beam_search.score_fn_last,
#        nbeams=nbeams,
#        length=length,
#        factor=factor,
#        intermediates=False)
#    print(all_beams[0].handlers.observed_likelihood)
#    print(last_beams[0].handlers.observed_likelihood)
#    # Note: Probabilistically true, but not strictly true in all worlds.
#    self.assertGreater(all_beams[0].handlers.observed_likelihood,
#                     last_beams[0].handlers.observed_likelihood)


if __name__ == '__main__':
  absltest.main()
