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

"""Tests for interpreter."""

import math

from absl.testing import absltest
from cascades._src import handlers
from cascades._src import handlers_test
from cascades._src import interpreter
from numpyro import distributions as np_dists


class LikelihoodTest(absltest.TestCase):

  def test_likelihood_weighting(self):
    """Sample from normal, and weight using mixture."""
    locs = [-5.0, 5.0]

    def _fn(verbose=False):
      del verbose
      return handlers_test.gaussian_mixture_likelihood(mixture_locs=locs)

    mixture = handlers_test._gaussian_mixture(locs)
    tracer = interpreter.Interpreter(_fn, seed=0)
    self.assertFalse(tracer._started)
    tracer.run()
    self.assertTrue(tracer._started)
    expected_score = mixture.log_prob(tracer['return_value'])

    self.assertAlmostEqual(expected_score, tracer.stats['observed_likelihood'])


class RejectTest(absltest.TestCase):

  def test_reject(self):

    def _reject_test():
      yield handlers.log('log1 should trigger', 'Log 1')
      yield handlers.reject(reason='rejected for no reason')
      yield handlers.log('log2 should not trigger', 'Log 2')

    tracer = interpreter.Interpreter(_reject_test, seed=0)
    tracer.run()
    self.assertTrue(math.isinf(tracer.stats['observed_likelihood']))
    self.assertLess(tracer.stats['observed_likelihood'], 0)
    self.assertIsInstance(tracer[list(tracer.tape)[-2]], handlers.Reject)
    self.assertLen(tracer.tape, 3)


class ObserveTest(absltest.TestCase):

  def test_observe(self):

    def foo():
      dist = np_dists.Normal()
      a = yield handlers.sample(name='a', dist=dist)
      b = yield handlers.sample(name='b', dist=dist)
      c = yield handlers.sample(name='c', dist=dist)
      return a + b + c

    tracer = interpreter.Interpreter(foo, observed=dict(a=100, c=-100))
    tracer.run()

    self.assertEqual(100, tracer['a'].value)
    self.assertIsInstance(tracer['a'], handlers.Observe)

    self.assertIsInstance(tracer['b'], handlers.Sample)

    self.assertEqual(-100, tracer['c'].value)
    self.assertIsInstance(tracer['c'], handlers.Observe)


class AutonameTest(absltest.TestCase):

  def test_observe(self):

    def foo():
      dist = np_dists.Normal()
      a = yield handlers.sample(name='a', dist=dist)
      b = yield handlers.sample(name='a', dist=dist)
      c = yield handlers.sample(name='a', dist=dist)
      return a + b + c

    tracer = interpreter.Interpreter(foo, observed={'a': 100, 'a/2': -100})
    for v in tracer._observed_used.values():
      self.assertFalse(v)
    tracer.run()

    self.assertEqual(100, tracer['a'].value)
    self.assertEqual(-100, tracer['a/2'].value)
    self.assertIsInstance(tracer['a'], handlers.Observe)
    self.assertIsInstance(tracer['a/1'], handlers.Sample)
    self.assertIsInstance(tracer['a/2'], handlers.Observe)


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
    self.assertEqual(param, tracer['test_param'].value)


if __name__ == '__main__':
  absltest.main()
