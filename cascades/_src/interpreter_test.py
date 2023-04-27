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

"""Tests for interpreter."""

from absl.testing import absltest
from cascades._src import handlers
from cascades._src import interpreter
from numpyro import distributions as np_dists


class InjectObservationHook(interpreter.InferenceHook):
  """A simple hook that injects observations into a `sample` call."""

  def __init__(self, observed):
    self._observed = observed

  def handle_sample(self, effect, rng, stats):
    effect.kwargs['rng'] = rng
    if effect.name in self._observed:
      effect.value = self._observed[effect.name]
    else:
      random_sample = handlers.dists.sample_distribution(
          effect.fn, *effect.args, **effect.kwargs)
      effect.value = random_sample.value
      effect.score = random_sample.log_p
    return effect


class AutonameTest(absltest.TestCase):

  def test_observe(self):

    def foo():
      dist = np_dists.Normal()
      a = yield handlers.sample(name='a', dist=dist)
      b = yield handlers.sample(name='a', dist=dist)
      c = yield handlers.sample(name='a', dist=dist)
      return a + b + c

    simple_hook = InjectObservationHook(observed={'a': 100, 'a/2': -100})

    tracer = interpreter.Interpreter(foo, inference_hook=simple_hook)
    tracer.run()

    self.assertEqual(100, tracer['a'].value)
    self.assertEqual(-100, tracer['a/2'].value)


class ParamTest(absltest.TestCase):

  def test_param(self):
    # Example of a simple parameter store
    def param_test():
      param_value = yield handlers.param(name='test_param')
      return param_value

    param = 'TEST PARAMETER'
    tracer = interpreter.Interpreter(
        param_test, param_store=dict(test_param=param))
    tracer.run()
    self.assertEqual(param, tracer.tape['test_param'].value)


class SamplesTest(absltest.TestCase):

  def test_project(self):
    full_data = [dict(a=1, b=17, c=-1), dict(a=2, b=88, c=-42)]
    full_samples = interpreter.Samples(full_data[0].keys(), full_data)
    projected_a = full_samples.project(['a'])
    self.assertEqual(dict(a=1), projected_a.get_row(0))
    self.assertEqual(dict(a=2), projected_a.get_row(1))
    self.assertEqual(projected_a.size(), full_samples.size())
    self.assertEqual(['a'], list(projected_a._variable_names.keys()))

    projected_bc = full_samples.project(['b', 'c'])
    self.assertEqual(projected_bc.size(), full_samples.size())
    self.assertEqual(['b', 'c'], list(projected_bc._variable_names.keys()))


if __name__ == '__main__':
  absltest.main()
