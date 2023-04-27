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

"""Runs an inferencer over all of the elements in a dataset."""

from cascades._src import interpreter
from cascades._src.distributions import base as dists
from cascades._src.inference import base
import jax
import tqdm


class DatasetSampler(base.Inferencer):
  """Runs an inferencer over all of the elements in a dataset."""

  def __init__(self, inferencer_class, **kwargs):
    self._inferencer_class = inferencer_class
    self._kwargs = kwargs

  def sample(self, dataset, num_samples=1, seed=0):
    rng = dists.get_rng(seed)
    samples = interpreter.Samples()
    self.debug_info = []
    for inputs in tqdm.tqdm(dataset):
      rng, subrng = jax.random.split(rng)
      these_kwargs = dict(self._kwargs)
      these_kwargs.update(inputs)
      inferencer = self._inferencer_class(**these_kwargs)
      these_samples = inferencer.sample(seed=subrng, num_samples=num_samples)
      samples.update(these_samples)
      self.debug_info.append(inferencer.debug_info)
    return samples
