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

"""Beam search on the observed likelihood of a program.

TODO(ddohan):
- Deduplicate this Tracer with forward sampler in core.Tracer.
- Allow parallel expansions using a threadpool.
- replace Factor with ValueFunction(callabe)


Consider separate score function:
- beam_search_by_score(score_fn=all_scores)
- beam_search_by_score(score_fn=last_score)


"""
import dataclasses
from typing import Any, Callable, Dict, Generator, List, Optional, Text

from cascades._src import handlers as h
import jax
import shortuuid


Trace = Dict[Text, h.Effect]
# yield, send, return types
TraceGenerator = Generator[h.Effect, Any, Any]


def run_until_observe(gen: TraceGenerator):
  """Run generator until next observe statement."""

  while True:
    effect = next(gen)
    if isinstance(effect, h.Observe):
      return effect


def clone(fn, trace: Trace, seed, param_handler=None):
  """Clone trace of function with a new seed.

  Args:
    fn: Function to trace.
    trace: Trace of site name -> Effect
    seed: Seed for clone of trace.
    param_handler: Effect handler to map parameter names to values.

  Returns:
    New handler stack instantiated around the function.
  """
  # Allow context handlers within program.
  handler_stack = [
      # Record effects to a tape
      h.Record,
      param_handler,
      # Sample/observe distributions & await futures
      h.Sampler,
      h.Observer,
  ]
  handler_fn = h.compose_handlers(handler_stack)
  handlers: h.Record = handler_fn(
      # Replace the known values when generator is rerun
      h.Replay(
          # Thread RNG through effect.
          gen_fn=h.Seed(seed=seed, gen_fn=fn),
          trace=trace,
          replay_scores=True,
          assert_all_used=True,
      )
  )
  return handlers


@dataclasses.dataclass
class Tracer:
  """Track state of tracers."""
  handlers: h.Record
  fn: Callable[[Any], Any]
  gen: TraceGenerator
  seed: Any
  parent: Optional[Any] = dataclasses.field(repr=False, default=None)
  result: Optional[Any] = None
  done: bool = False

  uid: Text = dataclasses.field(
      repr=True, default_factory=lambda: shortuuid.uuid()[:8])

  @classmethod
  def new(cls, fn, seed, param_handler=None):
    handlers = clone(
        fn=fn, trace=dict(), seed=seed, param_handler=param_handler)
    return Tracer(handlers=handlers, fn=fn, gen=handlers(), seed=seed)

  def __hash__(self):
    return hash(self.uid)

  @property
  def trace(self):
    """Get traces."""
    return self.handlers.trace

  def reify(self):
    """Run tracer to completion."""
    while not self.done:
      self.run_until_observe()

  def clone(self, seed, fastforward=True):
    """Make a clone of the tracer.

    Args:
      seed: Seed for the tracer. Only applies past the end of the copied trace.
        This is because the sites copied from the parent tracer are injected
        from the replayed trace. Only locations not included in the original
        trace make use of the new RNG seed.
      fastforward: if True, then fastforward the tracer to end of existing trace
        before returning.

    Returns:
      Cloned Tracer instance.
    """
    handlers = clone(fn=self.fn, trace=self.trace, seed=seed)
    tracer = Tracer(
        handlers=handlers, fn=self.fn, gen=handlers(), seed=seed, parent=self)

    if fastforward:
      # Fastforward clone to same location.
      for _ in range(len(self.handlers.trace)):
        next(tracer.gen)
    return tracer

  def run_until_observe(self):
    try:
      return run_until_observe(self.gen)
    except StopIteration as e:
      if e.args is not None and len(e.args):
        self.result = e.args[0]
      self.done = True


def expand(tracer: Tracer, k: int) -> List[Tracer]:
  """Expand a tracer to `k` children.

  Given tracer is cloned `k-1` times and then it and its children are run until
  next observe.

  Args:
    tracer: Tracer to expand.
    k: Number of expansions of the given tracer.

  Returns:
    List of tracers consisting of given tracer and its k-1 children, all
    advanced for 1 step.
  """
  seed = tracer.seed
  if isinstance(seed, int):
    seed = jax.random.PRNGKey(seed)
  seeds = jax.random.split(seed, num=k - 1)
  tracers = [tracer]
  for seed in seeds:
    # Clone the tracer k-1 times and fastforward each to same location
    child = tracer.clone(seed=seed, fastforward=True)
    tracers.append(child)

  for tracer in tracers:
    tracer.run_until_observe()
  return tracers


def score_fn_last(tracer):
  record: h.Record = tracer.handlers
  most_recent_key: Text = record.keys[-1]
  most_recent: h.Effect = record.trace[most_recent_key]
  if not isinstance(most_recent, h.Observe):
    raise ValueError(
        'last_score_fn should be called immediately after an Observe')
  return -most_recent.score


def score_fn_all(tracer):
  return -tracer.handlers.observed_likelihood


def beam_search_by_score(fn,
                         seed,
                         score_fn=score_fn_last,
                         nbeams=5,
                         expansion_factor=3,
                         max_rounds=20,
                         return_intermediates=False):
  """Beam search on a function prioritized by score_fn.

  TODO(ddohan): Reconsider how to handle completed traces when some may
    terminate earlier than others.

  Args:
    fn: Cascade function to run.
    seed: Seed for this run.
    score_fn: Method to use to prioritize beams. `last_score_fn` prioritizes by
      the last Observe and is appropriate for cost-to-go value functions, while
      `all_score_fn` prioritizes by sum of observed likelihoods and is
      appropriate for e.g. sampling from a language model using sum of
      likelihoods as the score.
    nbeams: Number of beams in active set.
    expansion_factor: Number of expansions per beam in each step.
    max_rounds: Maximum length of trajectories. Used to prevent runaway
      generations.
    return_intermediates: If True, return a list of intermediate tracers.

  Returns:
    If `return_intermediates` is false, list of highest scoring tracers. Else,
    return dict with `intermediates` set and `completed` tracers.
  """
  completed_traces = []
  intermediates = set()

  seeds = jax.random.split(jax.random.PRNGKey(seed), nbeams)
  active_traces = [Tracer.new(fn=fn, seed=seed) for seed in seeds]

  def step(tracers):
    expansions = []
    for root in tracers:
      root_expansions = expand(root, k=expansion_factor)
      expansions.extend(root_expansions)
    return sorted(expansions, key=score_fn)

  for _ in range(max_rounds):
    if len(completed_traces) >= nbeams:
      if return_intermediates:
        return dict(intermediates=intermediates, completed=completed_traces)
      else:
        return completed_traces

    expansions = step(active_traces)
    if return_intermediates:
      new = 0
      for tracer in expansions:
        if tracer not in intermediates:
          new += 1
        intermediates.add(tracer)

    active_traces = []
    for tracer in expansions:
      if len(completed_traces) >= nbeams:
        break
      if tracer.done:
        completed_traces.append(tracer)
      else:
        if len(active_traces) >= nbeams:
          break
        # TODO(ddohan): Consider adding additional clones if we run out of
        # pending traces before getting `nbeams` active traces.
        active_traces.append(tracer)
