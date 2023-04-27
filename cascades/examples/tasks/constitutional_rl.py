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

"""Constitutional RL dataset."""
import gzip
import json
import os


import gzip
open_file = gzip.open
DATA_PATH = None

VARIANTS = [
    'helpful-base', 'helpful-online', 'harmless-base',
    'helpful-rejection-sampled', 'red-team-attempts'
]


def read_gz_jsonl(path):
  """Read a compressed jsonl.gz file.

  Args:
    path: Path to load

  Returns:
    Tuple of (loaded dicts, lines which gave decode errors.)

  """
  with open_file(path, 'rb') as f:
    print(path)
    if path.endswith('.gz'):
      f = gzip.GzipFile(fileobj=f)

    xs = []
    errors = []
    for _, line in enumerate(f):
      line = line.decode()
      try:
        example = json.loads(line)
        xs.append(example)
      except json.JSONDecodeError:
        errors.append(line)
  return xs, errors


def load_dataset(variant='helpful-base', base_dir=DATA_PATH):
  """Load Constitutional RL data.


  Available at:
    https://github.com/anthropics/hh-rlhf

  Args:
    variant: Dataset subset to load.
    base_dir: Directory containing `train.jsonl` and `test.jsonl`.

  Returns:
    A dictionary of (train, test).
  """
  base_dir = os.path.join(base_dir, variant)
  if variant == 'red-team-attempts':
    train_path = os.path.join(base_dir, 'red_team_attempts.jsonl.gz')
    train_xs, _ = read_gz_jsonl(train_path)
    return dict(train=train_xs)
  else:
    train_path = os.path.join(base_dir, 'train.jsonl.gz')
    test_path = os.path.join(base_dir, 'test.jsonl.gz')

    train_xs, _ = read_gz_jsonl(train_path)
    test_xs, _ = read_gz_jsonl(test_path)
    return dict(train=train_xs, test=test_xs)

