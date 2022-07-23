# Language Model Cascades

The Cascades paper is [available on arXiv](https://arxiv.org/abs/2207.10342).

## Abstract

Prompted models have demonstrated impressive few-shot learning abilities. Repeated interactions at test-time with a single model, or the composition of multiple models together, further expands capabilities. These compositions are probabilistic models, and may be expressed in the language of graphical models with random variables whose values are complex data types such as strings. Cases with control flow and dynamic structure require techniques from probabilistic programming, and allow implementing disparate model structures and inference strategies in a unified language. We describe several existing techniques from this perspective, including scratchpads and chain of thought, verifiers, STaR, selection-inference, and tool use. We refer to the resulting programs as _language model cascades_.

## Overview

Large Language Models (LLMs) like GPT-3, LaMDA, and PaLM continue to benefit from scale but see diminishing returns as datasets are used up and models become harder and harder to serve. Recent progress has come not just from scale but from chaining LLMs together: 

* Scratchpad/Chain of Thought prompting allows language models to write intermediate results/reasoning to a shared buffer.
* Web-GPT and TALM call external tools, like calculators or search engines, to ground their results.
* STAR bootstraps its reasoning ability by iteratively solving and fine-tuning on its own samples.

We coin the term [Language Model Cascades](https://arxiv.org/abs/2207.10342) to describe a probabilistic programming language (PPL) and framework for expressing computer programs that chain together (or cascade) language models interacting with themselves, each other, and with external tools. We show that many recent papers can be expressed in this framework, which makes it easy to sample from cascaded programs and performance Bayesian optimization and inference strategies, like SMC, rejection sampling, and variational inference.

![examples of LLM cascades](https://github.com/google-research/cascades/blob/web/_data/cascades.png)

## Probabilistic Programming with LLMs

"“Probabilistic programs are usual functional or imperative programs with two added constructs: 
(1) the first is the ability to draw values at random from distributions
(2) the second being the ability to condition values of variables in a program via observations.”"

Probabilistic programs are different from traditional programming languages because they equip them with the ability to sample from distributions and observe variables based on data. This allows us to make predictions conditioned on certain inputs or outputs of a program or network. For example, we can sample prompts conditioned on the output of a verifier or external tool.

Cascades is unique in providing a probabilistic programming framework over the space of strings. Language models take in and emit text written in language. Cascades lets us perform various kinds of conditional and unconditional inference over this space.

## Implementatiion

The core library implementation is available on Github at [google-research/cascades](https://github.com/google-research/cascades). Examples of common cascades, using publicly available GPT models, will be released in coming weeks.

![The Cascades paper](https://github.com/google-research/cascades/blob/web/_data/cascades-paper.png)
