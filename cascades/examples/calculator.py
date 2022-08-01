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

"""Utilities for reasoning with a calculator."""
import ast
import operator as op

# https://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string
# supported operators
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg
}


def eval_arithmetic(expr):
  """Eval an arithmetic expression.

    >>> eval_expr('2^6')
    4
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0

  Args:
    expr: Expression to evaluate

  Returns:
    Result of evaluating expression.
  """
  return _eval_arithmetic(ast.parse(expr, mode='eval').body)


def _eval_arithmetic(node):
  if isinstance(node, ast.Num):  # <number>
    return node.n
  elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
    return operators[type(node.op)](_eval_arithmetic(node.left),
                                    _eval_arithmetic(node.right))
  elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
    return operators[type(node.op)](_eval_arithmetic(node.operand))
  else:
    raise TypeError(node)
