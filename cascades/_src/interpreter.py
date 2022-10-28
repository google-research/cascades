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

"""Default interpreter for cascade models.

Contains an implementation of basic functionality in a single effect
handler. Intended as a safe default interpreter and for easy modification for
custom functionality.
"""
import collections
import dataclasses
import math
from typing import Any, Dict, Optional

from cascades._src import handlers
import jax


def setup_seed(key_or_seed):
  if isinstance(key_or_seed, int):
    return jax.random.PRNGKey(seed=key_or_seed)
  else:
    return key_or_seed


def tape_to_dict(tape, stats):
  """Converts a tape from the interpreter to a map from variables to values."""
  result = dict()
  for name, effect in tape.items():
    if (isinstance(effect, handlers.Sample) or
        isinstance(effect, handlers.Observe) or
        isinstance(effect, handlers.Log)):
      result[name] = effect.value
  for key, value in dataclasses.asdict(stats).items():
    converted_key = '_' + str(key)
    result[converted_key] = value
  return result


def mode(list_):
  return collections.Counter(list_).most_common(1)[0][0]


class Samples():
  """Set of samples produced from an inference algorithm."""

  def __init__(self, variable_names=None, data=None):
    self._variable_names = variable_names or collections.OrderedDict()
    self._data = data or list()

  def append_row(self, variables_dict):
    self._variable_names.update({k: None for k in variables_dict.keys()})
    this_data = {
        name: variables_dict.get(name, None) for name in self._variable_names}
    self._data.append(this_data)

  def update_row(self, variables_dict):
    """Updates the most recent row based on the values in variables_dict."""
    self._variable_names.update({k: None for k in variables_dict.keys()})
    this_data = self._data[-1]
    this_data.update(variables_dict)

  def get_row(self, ri):
    return dict(self._data[ri])

  def get_column(self, column_name):
    if column_name not in self._variable_names:
      raise KeyError(column_name)
    return [d.get(column_name, None) for d in self._data]

  @property
  def data(self):
    return self._data

  @property
  def columns(self):
    return self._variable_names.keys()

  def size(self):
    return len(self._data)

  def update(self, other):
    # `other` is of the same class, so this is OK.
    # pylint: disable=protected-access
    for ri in range(len(other._data)):
      self.append_row(other._data[ri])
    # pylint: enable=protected-access

  def project(self, subvars):
    variable_names = collections.OrderedDict([(k, None) for k in subvars])
    rows = []
    for ri in range(len(self._data)):
      datum = self.get_row(ri)
      rows.append({k: datum.get(k, None) for k in variable_names.keys()})
    return Samples(variable_names, rows)

  def group_by(self, column_name):
    subgroups = collections.defaultdict(list)
    for ri in range(len(self._data)):
      key = self._data[ri][column_name]
      subgroups[key].append(self._data[ri])
    return [Samples(self._variable_names, v) for v in subgroups.values()]


@dataclasses.dataclass
class Stats():
  likelihood_sampled: float = 0.0
  likelihood_observed: float = 0.0


class InferenceHook():
  """Callbacks for the interpreter to implement inference algorithms."""

  def __init__(self, await_timeout=60):
    self._await_timeout = await_timeout

  def handle_sample(self, effect, rng, stats):
    raise NotImplementedError()

  def handle_observe(self, effect, rng, stats):
    raise NotImplementedError()

  def handle_log(self, effect, rng, stats):
    del rng, stats
    # Do nothing by default. It's OK not to handle a log.
    return effect


class Interpreter(handlers.EffectHandler):
  """Default effect handler for interpreting a cascades program.

  Example for a simple random walk from a parameterized location:
  ```
  def random_walk():
    # 2 step random walk starting at `start` and ending at 0.
    start = yield cc.param(name='start')
    step1 = yield cc.sample(Normal(loc=start), name='step1')
    yield cc.sample(Normal(loc=start + step1), obs=0.0, name='step2')

  tracer = cc.Interpreter(random_walk, param_store=dict(start=2.0))
  tracer.run()
  ```
  """

  def __init__(self,
               gen_fn,
               seed: int = 0,
               param_store: Optional[Dict[str, Any]] = None,
               await_timeout: int = 60,
               verbose: bool = False,
               autoname: bool = True,
               inference_hook: Optional[InferenceHook] = None):
    """Instantiate a cascades interpreter.

    Args:
      gen_fn: Callable generator.
      seed: Random number seed.
      param_store: Map from parameter name to value.
      await_timeout: Length of time to wait for futures.
      verbose: If True, log effects.
      autoname: If True, automatically increments a unique id each time a
        variable name reappears.
      inference_hook: Called when events are received. This is useful for
        implementing inference algorithms.
    """
    super().__init__(gen_fn=gen_fn)
    self._rng = setup_seed(seed)
    self._tape = dict()
    self._param_store = param_store or dict()
    self._await_timeout = await_timeout
    self._verbose = verbose
    self._likelihood_sampled = 0.0
    self._likelihood_observed = 0.0
    self._autoname = autoname
    self._name_counts = collections.defaultdict(int)
    self._exception = None
    self._inference_hook = inference_hook
    self._stats = Stats()

    # Track if this interpreter has been started already.
    self._started = False

  def run(self):
    if self._started:
      raise ValueError(
          '`run` may only be called once per interpreter instance.')
    self._started = True
    gen = self()
    for eff in gen:
      if eff.should_stop:
        break
      pass
    return self

  def __getitem__(self, key):
    """Get item from trace tape by name."""
    return self._tape[key]

  @property
  def stats(self):
    return self._stats

  def __repr__(self):
    kvs = [f'    {k}: {v}' for k, v in self._tape.items()]
    kvs = ',\n'.join(kvs)
    formatted = str(dict(stats=self.stats))
    tape = f'{{\n{kvs}\n}}'
    return formatted[:-1] + f",\n 'tape': {tape}}}"

  def process(self, effect):
    self._rng, subrng = jax.random.split(self._rng, 2)

    def log(string):
      if self._verbose:
        print(string)

    if not effect.name:
      raise ValueError(f'Must name effect: {effect}')
    if effect.name in self._tape:
      if self._autoname:
        self._name_counts[effect.name] += 1
        idx = self._name_counts[effect.name]
        effect.name = f'{effect.name}/{idx}'
      else:
        raise ValueError(f'Effect name is not unique: {effect.name}')

    if isinstance(effect, handlers.Log):
      # Logs will get added to the tape. Also ask the inference_hook
      # if it wants to do anything.
      if self._inference_hook:
        effect = self._inference_hook.handle_log(effect, subrng, self._stats)
    elif isinstance(effect, handlers.Reject):
      effect.score = -jax.numpy.inf
      self._stats.likelihood_observed = -jax.numpy.inf
    elif isinstance(effect, handlers.Param):
      # aka handlers.ParamStore
      log('Param encountered: {effect}')
      if effect.name in self._param_store:
        log(f'Found parameter in store: {effect}')
        effect.value = self._param_store[effect.name]
      else:
        if effect.value is None:
          # Initialize by sampling from the distribution
          log(f'Sampling new param value: {effect.name}')
          if not effect.fn:
            raise ValueError(f'Param fn cannot be None: `{effect}`')
          random_sample = handlers.dists.sample_distribution(
              fn=effect.fn, *effect.args, **effect.kwargs)
          effect.value = random_sample.value
          effect.log_p = random_sample.log_p
        self._param_store[effect.name] = effect.value
    elif isinstance(effect, handlers.Sample):
      if self._inference_hook:
        effect = self._inference_hook.handle_sample(effect, subrng, self._stats)
    elif isinstance(effect, handlers.Observe):
      if self._inference_hook:
        effect = self._inference_hook.handle_observe(effect, subrng,
                                                     self._stats)

    if effect.score is not None:
      # Stop on Reject and infinite likelihoods
      should_stop = jax.numpy.any(jax.numpy.isinf(effect.score))
      if jax.device_get(should_stop):
        effect.should_stop = True

    # Record to a tape. aka handlers.Record
    self._tape[effect.name] = effect
    return effect

  def on_return(self, return_value):
    if 'return_value' in self._tape:
      raise ValueError(
          f'Cannot have `return_value` already recorded in tape: {self._tape.keys()}'
      )
    self._tape['return_value'] = return_value

  def __call__(self, fn_or_gen=None, nested=False):
    """Improved exception handling around the base handler."""
    try:
      return_value = yield from super().__call__(
          fn_or_gen=fn_or_gen, verbose=self._verbose, nested=nested)
      return return_value
    except Exception as e:
      self._exception = e
      print('Caught exception')
      print(e)
      self._tape['exception'] = e
      self._likelihood_observed += -math.inf
      # TODO(ddohan): Maintain more traceback context!
      raise e

  @property
  def tape(self):
    return self._tape


def try_sample(model, capture=False, **kwargs) -> Interpreter:
  """Sample from model, with error capture.

  Args:
    model: Cascade model fn to sample.
    capture: If True, then capture and log exceptions.
    **kwargs: Arguments passed to Interpreter

  Returns:
    Interpreter instance after running through program.
  """
  tracer = Interpreter(model, **kwargs)
  try:
    tracer.run()
  except Exception as e:  # pylint: disable=broad-except
    if capture:
      print(f'Caught exception: {e}')
    else:
      raise e

  return tracer
