"""
modified from
"""
from collections import Counter
import typing

import torch
import torchmetrics as tm

r"""
BLEU = BP * exp(1/n * sum_{n=1}^N(log(p_i)))
BP = \begin{cases} 1 & if c > r \\ exp(1 - r/c) & if c <= r \end{cases}

c: length of candidate
r: length of reference
p_i: precision of i-gram
N: n-gram order
"""


def _count_ngram(ngram_input_list: typing.Sequence[str], n_gram: int) -> Counter:
    """Counting how many times each word appears in a given text with ngram.

    Args:
        ngram_input_list: A list of translated text or reference texts
        n_gram: gram value ranged 1 to 4

    Return:
        ngram_counter: a collections.Counter object of ngram
    """

    ngram_counter: Counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(len(ngram_input_list) - i + 1):
            ngram_key = tuple(ngram_input_list[j: (i + j)])
            ngram_counter[ngram_key] += 1

    return ngram_counter


class BLEU(tm.Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(
            self,
            n_gram: int = 4,
            tokenize_fn: typing.Optional[typing.Callable] = None,
            smooth: bool = False,
            weights: typing.Optional[typing.Sequence[float]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_gram = n_gram
        self.tokenize_fn = tokenize_fn
        if tokenize_fn is None:
            self.tokenize_fn = lambda x: x.split()
        self.smooth = smooth
        if weights is not None and len(weights) != n_gram:
            raise ValueError(f"List of weights has different weights than `n_gram`: {len(weights)} != {n_gram}")
        self.weights = weights if weights is not None else [1.0 / n_gram] * n_gram

        self.add_state("preds_len", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_len", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numerator", torch.zeros(self.n_gram), dist_reduce_fx="sum")
        self.add_state("denominator", torch.zeros(self.n_gram), dist_reduce_fx="sum")

    def update(
            self,
            preds: typing.Sequence[str],
            target: typing.Sequence[typing.Union[str, typing.Sequence[str]]]
    ):
        preds = [preds] if isinstance(preds, str) else preds
        target = [[tgt] if isinstance(tgt, str) else tgt for tgt in target]

        preds_words = [self.tokenize_fn(line) if line else [] for line in preds]
        target_words = [[self.tokenize_fn(line) for line in tgt] for tgt in target]

        for p, t in zip(preds_words, target_words):
            self.preds_len += len(p)
            target_len_list = [len(tgt) for tgt in t]
            target_len_diff = [abs(len(p) - tgt_len) for tgt_len in target_len_list]
            self.target_len += target_len_list[target_len_diff.index(min(target_len_diff))]

            preds_counter = _count_ngram(p, self.n_gram)
            target_counter = Counter()
            for _t in t:
                target_counter |= _count_ngram(_t, self.n_gram)

            ngram_counter_clip = preds_counter & target_counter

            for counter_clip in ngram_counter_clip:
                self.numerator[len(counter_clip) - 1] += ngram_counter_clip[counter_clip]

            for counter in preds_counter:
                self.denominator[len(counter) - 1] += preds_counter[counter]

    def compute(self):
        device = self.numerator.device
        if min(self.numerator) == 0.0:
            return torch.tensor(0.0, device=device)

        if self.smooth:
            precision = torch.div(
                self.numerator + torch.ones(self.n_gram, device=device),
                self.denominator + torch.ones(self.n_gram, device=device)
            )
            precision[0] = self.numerator[0] / self.denominator[0]
        else:
            precision = self.numerator / self.denominator

        log_precision = torch.tensor(self.weights, device=device) * torch.log(precision)
        geometric_mean = torch.exp(torch.sum(log_precision))

        if self.preds_len > self.target_len:
            brevity_penalty = torch.tensor(1.0, device=device)
        else:
            brevity_penalty = torch.exp(1 - self.target_len / self.preds_len)

        return brevity_penalty * geometric_mean
