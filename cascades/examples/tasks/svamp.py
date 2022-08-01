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

"""SVAMP word problem dataset."""
import json
import random
import re
from typing import Text, Optional, Union

import cascades as cc
from cascades.examples import calculator

open_file = open
SVAMP_PATH = None


def load_svamp(seed=0, valid_size=50, test_size=50, json_path=SVAMP_PATH):
  """Load SVAMP dataset from json.

  Available at:
    https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json

  Args:
    seed: Seed for shuffling.
    valid_size: Number of examples in validation set.
    test_size: Number of examples in test set.
    json_path: Json file path to load.

  Returns:
    A dictionary of (train, valid, test).
  """

  if not json_path:
    raise ValueError('Missing json_path for load_svamp must be specified.')

  with open_file(json_path) as f:
    dataset = json.loads(f.read())
  random.Random(seed).shuffle(dataset)
  return dict(
      valid=dataset[:valid_size],
      test=dataset[valid_size:valid_size + test_size],
      train=dataset[valid_size + test_size:])


def _evaluate_equation(text: Text) -> Optional[float]:
  try:
    output = calculator.eval_arithmetic(text)
    return output
  except (AttributeError, SyntaxError, ZeroDivisionError) as e:
    del e
    return None


def _score_prediction(model_answer: Union[float, int, Text],
                      true_answer: Union[float, int, Text]):
  model_answer = str(model_answer)
  true_answer = str(true_answer)
  if true_answer == model_answer:
    return True
  else:
    return False


def _svamp_format_task(task):
  return '{body} : {question}'.format(
      body=task['Body'], question=task['Question'])


def _text_extract_regex(text: Text, pattern: Text) -> Optional[Text]:
  p = re.compile(pattern)
  matches = p.findall(text)
  if not matches:
    return None
  matched_text = matches[0]
  return matched_text


def _text_extract_brackets(text: Text) -> Optional[Text]:
  """Extract text within {brackets}."""
  return _text_extract_regex(text, pattern=r'{{(.*?)}}')


def _svamp_format_prompt(
    task,
    examples,
    instructions='Write the arithmetic expression to solve each word problem:'):
  """Create prompt based on svamp task."""
  formatted_io_examples = [(_svamp_format_task(task),
                            '{{ %s }}' % task['Equation']) for task in examples]
  parts = [instructions]
  for question, answer in formatted_io_examples:
    parts.append('QUESTION: %s\nANSWER: %s' % (question, answer))

  parts.append('QUESTION: %s\nANSWER:' % _svamp_format_task(task))
  return '\n\n'.join(parts)


def solve_svamp_task(task, train_set, lm=None, check_correct=False):
  """Generate solution questions to the SVAMP task, conditioned on train_set."""
  # TODO(ddohan): Make distribution over conditioning questions a part of model.
  prompt = _svamp_format_prompt(task=task, examples=train_set)

  # Generate equations
  line = yield cc.sample(
      cc.String(prompt=prompt, lm=lm, until='\n'), name='gen_equation')

  # Clip to 1 line and rescore
  # line = line.strip().split('\n')[0]

  # TODO(ddohan): Should this go in the prior p(x) instead of likelihood p(y|x)
  # yield cc.observe(obs=line, dist=lm(prompt=prompt), name='clip_and_rescore')

  # Extract equations between {{ }}
  eq = _text_extract_brackets(line)
  eq = eq.strip()
  if eq is None:
    yield cc.reject(reason='No equation proposed', name='no_equation')
  eq = eq.replace('{', '').replace('}', '')

  # Evaluate the arithmetic using a calculator
  calculator_output = _evaluate_equation(eq)

  if calculator_output is None:
    yield cc.reject(reason=f'Calculator failed: `{eq}`')

  if check_correct:
    # Did we get the right answer?
    correct = _score_prediction(model_answer=calculator_output,
                                true_answer=task['Answer'])
    yield cc.log(correct, 'is_correct')
    if not correct:
      yield cc.reject(
          reason=f'Incorrect answer: expected {task["Answer"]} != {eq} from {calculator_output}',
          name='incorrect')

  return (eq, calculator_output)


# Maximize p(score | task) marginalized over solutions
# params = set of prompts we use in the prefix.
def svamp_optimize_prompt(lm, task, train_set, k=5):
  """Choose k tasks to put into prompt from training examples."""
  idxs = yield cc.sample(dist=cc.Choose(k=k, options=range(len(train_set))))  # pytype: disable=wrong-arg-types
  prompt_tasks = [train_set[i] for i in idxs]
  result = yield solve_svamp_task(lm=lm, task=task, train_set=prompt_tasks)
  return result
