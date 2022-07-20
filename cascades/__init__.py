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

"""Cascades API."""

# A new PyPI release will be pushed everytime `__version__` is increased
# When changing this, also update the CHANGELOG.md
__version__ = '0.2.0'

from cascades._src.distributions.base import UniformCategorical
from cascades._src.distributions.choose import Choose
from cascades._src.distributions.strings import get_default_lm
from cascades._src.distributions.strings import mock_lm
from cascades._src.distributions.strings import String
from cascades._src.handlers import factor
from cascades._src.handlers import log
from cascades._src.handlers import observe
from cascades._src.handlers import param
from cascades._src.handlers import reject
from cascades._src.handlers import rejection_sample
from cascades._src.handlers import sample
from cascades._src.sampler import Sampler
