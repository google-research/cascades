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

"""GSM8k reasoning task."""
import json
import os

open_file = open
GSM8K_PATH = None


def _load_line(line):
  example = json.loads(line)
  answer = int(example['answer'].split('#### ')[-1].replace(',', ''))
  example['final_answer'] = answer
  return example


def load_dataset(base_dir=GSM8K_PATH):
  """Load GSM8k raw data from `base_dir`.

  Available at:
    https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json

  Args:
    base_dir: Directory containing `train.jsonl` and `test.jsonl`.

  Returns:
    A dictionary of (train, test).
  """
  train_path = os.path.join(base_dir, 'train.jsonl')
  test_path = os.path.join(base_dir, 'test.jsonl')

  train_xs = []
  test_xs = []

  with open_file(train_path) as f:
    for i, line in enumerate(f):
      example = _load_line(line)
      example['id'] = f'train/{i}'
      train_xs.append(example)

  with open_file(test_path) as f:
    for i, line in enumerate(f):
      example = _load_line(line)
      example['id'] = f'test/{i}'
      test_xs.append(example)

  return dict(train=train_xs, test=test_xs)
