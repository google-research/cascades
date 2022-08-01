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

"""Scratchpad/chain of thought, demonstrated on gsm8k.

Scratchpad: https://arxiv.org/abs/2112.00114
Chain of thought: https://arxiv.org/abs/2201.11903

Self consistency (https://arxiv.org/abs/2203.11171) is done by marginalizing out
  the `thought` in the `infer` operator.
"""
import dataclasses
import re
from typing import Dict, Iterable, Tuple, Optional, Text, List
import cascades as cc
from cascades.examples.tasks import gsm8k

# Standard chain of thought arithmetic prompts.
# From https://arxiv.org/abs/2112.00114
PROMPTS = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.
"""


@dataclasses.dataclass
class ReasonIO:
  """Represents a question and answer, together with reasoning."""
  question: Text
  reason: Optional[Text] = None
  answer: Optional[Text] = None
  id: Optional[Text] = dataclasses.field(default=None)


# From https://arxiv.org/abs/2203.11171
def _process_part(qa: Text) -> ReasonIO:
  question, reason = qa.split('A:')
  question = question.replace('Q:', '')
  reason, answer = reason.split('The answer is')
  answer = answer.replace('.', '')
  return ReasonIO(
      question=question.strip(), reason=reason.strip(), answer=answer.strip())


def load_chain_examples() -> Tuple[ReasonIO]:
  """Load the standard chain of thought prompts used in the paper."""
  parts = PROMPTS.split('Q:')[1:]
  chain_prompts = tuple(_process_part(x) for x in parts)
  return chain_prompts


def load_gsm8k(base_dir=gsm8k.GSM8K_PATH):
  """Load and process the gsm8k dataset."""
  splits = gsm8k.load_dataset(base_dir=base_dir)
  train = tuple(map(process_gsm8k_example, splits['train']))
  test = tuple(map(process_gsm8k_example, splits['test']))
  return dict(train=train, test=test)


def process_gsm8k_example(x: Dict[Text, Text]) -> ReasonIO:
  """Convert gsm8k dicts into ReasonIO."""
  return ReasonIO(
      question=x['question'],
      reason=x['answer'].split('####')[0].strip(),
      answer=x['final_answer'],
      id=x['id'])


def fewshot_prompt(examples: Iterable[ReasonIO],
                   target: Optional[ReasonIO] = None):
  """Construct a few shot prompt."""
  parts = []
  for x in examples:
    if x.reason is None:
      raise ValueError('Must provide reason for few shot.')
    if x.answer is None:
      raise ValueError('Must provide answer for few shot.')
    part = f'Question: {x.question}\nReason: {x.reason}\nAnswer: {x.answer}'
    parts.append(part)
  if target:
    part = f'Question: {target.question}\nReason'
    parts.append(part)
  else:
    parts.append('')

  prompt = '\n===\n'.join(parts).strip()
  return prompt


def sample_with_prompts(target: ReasonIO,
                        examples: List[ReasonIO],
                        n_prompts: Optional[int] = None,
                        lm=None,
                        max_calls=4):
  """Sample an answer for target given prompts.

  Args:
    target: Target task.
    examples: Prompting tasks.
    n_prompts: If None, use all examples in prompt. Otherwise, select this many
      examples to place in prompt.
    lm: Language model to use.
    max_calls: Max # of times to iterate the LM sampling.

  Yields:
    Cascade distributions (all LM sample nodes in this case).

   Returns:
     predicted answer string
  """
  # Select a random subset and ordering of the prompts.
  yield cc.log(value=target.id, name='problem_id')
  if n_prompts:
    chosen_examples = yield cc.sample(
        cc.Choose(k=n_prompts, options=examples), name='choose_prompts')
  else:
    chosen_examples = examples

  # Create few shot prompt
  prompt = fewshot_prompt(examples=chosen_examples, target=target)

  # Sample until we hit the end of example marker (`===`), then extract
  # the answer as rightmost digits.
  prediction = yield cc.sample(
      cc.String(prompt=prompt, lm=lm, k=max_calls, until='==='), name='thought')

  # Find right most number.
  nums = re.findall(r'\d+', prediction)
  if nums:
    answer = nums[-1]
  else:
    yield cc.reject(reason=f'No answer found in `{prediction}`')
  return answer
