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

"""Loaders for bbh dataset: https://arxiv.org/abs/2210.09261."""
import json
import os  # pylint: disable=unused-import

open_file = open
list_dir = os.listdir
bbh_path = None
cot_prompts_path = None


def get_task_names():
  files = list_dir(bbh_path)
  task_names = [f.split('.json')[0] for f in files]
  task_names = [f for f in task_names if '.' not in f]
  return task_names


def get_cot_prompt(task_name: str):
  if task_name not in get_task_names():
    raise ValueError(
        f'Task {task_name} not a valid bbh task.  Consult `get_task_names()` for a list of valid tasks.'
    )

  prompt_loc = f'{cot_prompts_path}/{task_name}.txt'
  with open_file(prompt_loc, 'r') as f:
    data = ''.join(f.readlines())
  return data


def load_dataset(task_name: str,
                 base_dir: str = bbh_path,
                 qa_format: bool = True):
  """Load MATH raw data from disk.

  Available at:
    https://github.com/suzgunmirac/BIG-Bench-Hard

  Args:
    task_name: Which bbh task to load
    base_dir: Directory containing json files for bbh.
    qa_format: whether to prepend Q: and A: to example `input` and `target`

  Returns:
    A list of examples.
    Each example is a dict with keys:
    'input': (str) problem statement
    'target': (str) the solution
  """

  if task_name not in get_task_names():
    raise ValueError(
        f'Task {task_name} not a valid bbh task.  Consult `get_task_names()` for a list of valid tasks.'
    )

  task_loc = f'{base_dir}/{task_name}.json'
  with open_file(task_loc, 'r') as f:
    data = json.loads(f.readlines()[0])['examples']

  if qa_format:
    formatted_examples = []
    for d in data:
      # uses BIG-bench formatting
      formatted_examples.append({
          'input': f"Q: {d['input']}",
          'target': f"A: {d['target']}"
      })
    data = formatted_examples

  return data
