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

"""Loaders for MATH dataset."""
import functools
import json

open_file = open
MATH_PATH = None


def get_category(math_ds,
                 category='Prealgebra',
                 split='train',
                 length_sorted=False):
  tasks = [x for x in math_ds[split] if x['type'] == category]
  if length_sorted:
    tasks = sorted(tasks, key=lambda x: len(x['problem']) + len(x['solution']))
  return tasks


@functools.lru_cache()
def load_dataset(base_dir=MATH_PATH):
  """Load MATH raw data from disk.

  Available at:
    https://github.com/hendrycks/math/

  Args:
    base_dir: Directory containing `train.jsonl` and `test.jsonl`.

  Returns:
    A dictionary of (train, test).
    Each example is a dict with keys:
    'index': (int) example number
    'level': (int) difficulty
    'problem': (str) problem statement
    'solution': (str) an example solution
  """
  with open_file(base_dir, 'r') as f:
    data = json.loads(f.readlines()[0])

  return data
