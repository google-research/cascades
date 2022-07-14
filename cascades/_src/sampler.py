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

"""Sample values from a model."""
from functools import partial  # pylint: disable=g-importing-member
from typing import Any

from cascades._src import handlers as h
import jax

# Inference


# TODO(ddohan): Add goal conditioning (future prompting) back in.
def build_tracer(
    fn,
    seed: Any,  # h.RNGKey,
    trace=(),
    reparam_fn=None,
    rescore=False) -> h.Record:
  """Make a tracer which samples values and records to a tape.

  Args:
    fn: Function to trace.
    seed: Seed for clone of trace.
    trace: Trace of site name -> Effect to replay values.
    reparam_fn: Partial Reparam handler which can be used to reparameterize
      distributions.
    rescore: Whether to rescore the observed samples.

  Returns:
    New handler stack instantiated around the function.
  """
  # TODO(ddohan): Allow context handlers within program.
  # gen_fn = h.ApplyContext(fn)
  gen_fn = fn

  handler_stack = [
      # Record effects to a tape
      h.Record,
      # Sample/observe distributions & await futures
      h.StopOnReject,
      partial(h.Observer, rescore=rescore),
      h.Sampler,
      # Replace the known values when generator is rerun
      partial(h.Replay,
              # Thread RNG through effect.
              trace=trace,
              replay_scores=True,
              assert_all_used=True),
      partial(h.Seed, seed=seed),
  ]
  handler_fn = h.compose_handlers(handler_stack)
  if reparam_fn is None:
    reparam_fn = lambda x: x
  record_handler: h.Record = handler_fn(reparam_fn(gen_fn))
  return record_handler


class Sampler:
  """Run sampling on a model."""

  def __init__(
      self,
      model,
      # examples: Optional[Iterable[Dict[Text, Any]]] = None,
      # observe: Optional[Dict[Text, Any]] = None,
      # future_prompt=False,
      rescore=True,
      reparam_fn=None,
  ):
    self._model = model
    # self._examples = examples or ()
    # self._observed = observe or dict()
    # self._future_prompt = future_prompt
    self._rescore = rescore
    self._reparam_fn = reparam_fn

  def build_tracer(self, seed=0):
    """Instantiate a tracer for given seed. Does not evalute concrete values."""
    # TODO(ddohan): Handle observed variables, with and without future prompting
    tracer = build_tracer(
        self._model,
        seed=seed,
        # examples=self._examples,
        # observed=self._observed,
        # future_prompt=self._future_prompt,
        rescore=self._rescore,
        reparam_fn=self._reparam_fn)
    tracer.done = False
    return tracer

  def reify(self, seed=0):
    tracer = self.build_tracer(seed=seed)
    reify(tracer)
    return tracer

  def parallel(self, pool, seed, n):
    """Run function in parallel."""
    seeds = jax.random.split(jax.random.PRNGKey(seed), n)
    tracers = []
    for seed in seeds:
      tracer = self.build_tracer(seed=seed)
      f = pool.submit(reify, tracer=tracer)
      tracer.future = f
      tracers.append(tracer)
    return tracers


def reify(tracer: h.Record, verbose=False):
  """Run tracer and return the result. Makes implicit values explicit.

  Modifies `tracer` in place.

  Args:
    tracer: Tracer to run.
    verbose: If true, log out effects.

  Returns:
    Result of running the program.
  """
  it = tracer()
  return_value = None
  while True:
    try:
      eff = next(it)
      if verbose:
        print(eff)
    except StopIteration as e:
      if e.args is not None and len(e.args):
        return_value = e.args[0]
        if verbose:
          print(f'Return value: {eff}')
      break
  tracer.done = True
  tracer.return_value = return_value
  return return_value

