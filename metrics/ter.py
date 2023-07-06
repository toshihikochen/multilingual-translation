import typing
import re

import torch
import torchmetrics as tm

from torchmetrics.functional.text.helper import (
    _flip_trace,
    _LevenshteinEditDistance,
    _trace_to_alignment,
)

# Tercom-inspired limits
_MAX_SHIFT_SIZE = 10
_MAX_SHIFT_DIST = 50

# Sacrebleu-inspired limits
_MAX_SHIFT_CANDIDATES = 1000

r"""
score = edit(c, r) / len(r)
edit(c, r) = 1 - (levenshtein(c, r) / len(r))

levenshtein(c, r): edit distance between c and r
minimum number of single-character edits: insertions, deletions, or substitutions

c: candidate
r: reference
"""


def _find_shifted_pairs(pred_words: typing.List[str], target_words: typing.List[str]) -> typing.Iterator[
    typing.Tuple[int, int, int]]:
    """Find matching word sub-sequences in two lists of words. Ignores sub-sequences starting at the same position.

    Args:
        pred_words: A list of a tokenized hypothesis sentence.
        target_words: A list of a tokenized reference sentence.

    Return:
        Yields tuples of ``target_start, pred_start, length`` such that:
        ``target_words[target_start : target_start + length] == pred_words[pred_start : pred_start + length]``

        pred_start:
            A list of hypothesis start indices.
        target_start:
            A list of reference start indices.
        length:
            A length of a word span to be considered.
    """
    for pred_start in range(len(pred_words)):
        for target_start in range(len(target_words)):
            # this is slightly different from what tercom does but this should
            # really only kick in in degenerate cases
            if abs(target_start - pred_start) > _MAX_SHIFT_DIST:
                continue

            for length in range(1, _MAX_SHIFT_SIZE):
                # Check if hypothesis and reference are equal so far
                if pred_words[pred_start + length - 1] != target_words[target_start + length - 1]:
                    break
                yield pred_start, target_start, length

                # Stop processing once a sequence is consumed.
                _hyp = len(pred_words) == pred_start + length
                _ref = len(target_words) == target_start + length
                if _hyp or _ref:
                    break


def _handle_corner_cases_during_shifting(
        alignments: typing.Dict[int, int],
        pred_errors: typing.List[int],
        target_errors: typing.List[int],
        pred_start: int,
        target_start: int,
        length: int,
) -> bool:
    """A helper function which returns ``True`` if any of corner cases has been met. Otherwise, ``False`` is
    returned.

    Args:
        alignments: A dictionary mapping aligned positions between a reference and a hypothesis.
        pred_errors: A list of error positions in a hypothesis.
        target_errors: A list of error positions in a reference.
        pred_start: A hypothesis start index.
        target_start: A reference start index.
        length: A length of a word span to be considered.

    Return:
        An indication whether any of conrner cases has been met.
    """
    # don't do the shift unless both the hypothesis was wrong and the
    # reference doesn't match hypothesis at the target position
    if sum(pred_errors[pred_start: pred_start + length]) == 0:
        return True

    if sum(target_errors[target_start: target_start + length]) == 0:
        return True

    # don't try to shift within the subsequence
    if pred_start <= alignments[target_start] < pred_start + length:
        return True

    return False


def _perform_shift(words: typing.List[str], start: int, length: int, target: int) -> typing.List[str]:
    """Perform a shift in ``words`` from ``start`` to ``target``.

    Args:
        words: A words to shift.
        start: An index where to start shifting from.
        length: A number of how many words to be considered.
        target: An index where to end shifting.

    Return:
        A list of shifted words.
    """

    def _shift_word_before_previous_position(words: typing.List[str], start: int, target: int, length: int) -> \
            typing.List[str]:
        return words[:target] + words[start: start + length] + words[target:start] + words[start + length:]

    def _shift_word_after_previous_position(words: typing.List[str], start: int, target: int, length: int) -> \
            typing.List[str]:
        return words[:start] + words[start + length: target] + words[start: start + length] + words[target:]

    def _shift_word_within_shifted_string(words: typing.List[str], start: int, target: int, length: int) -> typing.List[
        str]:
        shifted_words = words[:start]
        shifted_words += words[start + length: length + target]
        shifted_words += words[start: start + length]
        shifted_words += words[length + target:]
        return shifted_words

    if target < start:
        return _shift_word_before_previous_position(words, start, target, length)
    if target > start + length:
        return _shift_word_after_previous_position(words, start, target, length)
    return _shift_word_within_shifted_string(words, start, target, length)


def _shift_words(
        pred_words: typing.List[str],
        target_words: typing.List[str],
        cached_edit_distance: _LevenshteinEditDistance,
        checked_candidates: int,
) -> typing.Tuple[int, typing.List[str], int]:
    """Attempt to shift words to match a hypothesis with a reference. It returns the lowest number of required
    edits between a hypothesis and a provided reference, a list of shifted words and number of checked candidates.

    Note that the filtering of possible shifts and shift selection are heavily based on somewhat arbitrary heuristics.
    The code here follows as closely as possible the logic in Tercom, not always justifying the particular design
    choices. (The paragraph copied from https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/lib_ter.py)

    Args:
        pred_words: A list of tokenized hypothesis sentence.
        target_words: A list of lists of tokenized reference sentences.
        cached_edit_distance: A pre-computed edit distance between a hypothesis and a reference.
        checked_candidates: A number of checked hypothesis candidates to match a provided reference.

    Return:
        best_score:
            The best (lowest) number of required edits to match hypothesis and reference sentences.
        shifted_words:
            A list of shifted words in hypothesis sentences.
        checked_candidates:
            A number of checked hypothesis candidates to match a provided reference.
    """
    edit_distance, inverted_trace = cached_edit_distance(pred_words)
    trace = _flip_trace(inverted_trace)
    alignments, target_errors, pred_errors = _trace_to_alignment(trace)

    best: typing.Optional[typing.Tuple[int, int, int, int, typing.List[str]]] = None

    for pred_start, target_start, length in _find_shifted_pairs(pred_words, target_words):
        if _handle_corner_cases_during_shifting(
                alignments, pred_errors, target_errors, pred_start, target_start, length
        ):
            continue

        prev_idx = -1
        for offset in range(-1, length):
            if target_start + offset == -1:
                idx = 0
            elif target_start + offset in alignments:
                idx = alignments[target_start + offset] + 1
            # offset is out of bounds => aims past reference
            else:
                break
            # Skip idx if already tried
            if idx == prev_idx:
                continue

            prev_idx = idx

            shifted_words = _perform_shift(pred_words, pred_start, length, idx)

            # Elements of the tuple are designed to replicate Tercom ranking of shifts:
            candidate = (
                edit_distance - cached_edit_distance(shifted_words)[0],  # highest score first
                length,  # then, longest match first
                -pred_start,  # then, earliest match first
                -idx,  # then, earliest target position first
                shifted_words,
            )

            checked_candidates += 1

            if not best or candidate > best:
                best = candidate

        if checked_candidates >= _MAX_SHIFT_CANDIDATES:
            break

    if not best:
        return 0, pred_words, checked_candidates
    best_score, _, _, _, shifted_words = best
    return best_score, shifted_words, checked_candidates


def _translation_edit_rate(pred_words: typing.List[str], target_words: typing.List[str]) -> torch.Tensor:
    """Compute translation edit rate between hypothesis and reference sentences.

    Args:
        pred_words: A list of a tokenized hypothesis sentence.
        target_words: A list of lists of tokenized reference sentences.

    Return:
        A number of required edits to match hypothesis and reference sentences.
    """
    if len(target_words) == 0:
        return torch.tensor(0.0)

    cached_edit_distance = _LevenshteinEditDistance(target_words)
    num_shifts = 0
    checked_candidates = 0
    input_words = pred_words

    while True:
        # do shifts until they stop reducing the edit distance
        delta, new_input_words, checked_candidates = _shift_words(
            input_words, target_words, cached_edit_distance, checked_candidates
        )
        if checked_candidates >= _MAX_SHIFT_CANDIDATES or delta <= 0:
            break
        num_shifts += 1
        input_words = new_input_words

    edit_distance, _ = cached_edit_distance(input_words)
    total_edits = num_shifts + edit_distance

    return torch.tensor(total_edits)


def _compute_sentence_statistics(pred_words: typing.List[str], target_words: typing.List[typing.List[str]]) -> \
        typing.Tuple[torch.Tensor, torch.Tensor]:
    """Compute sentence TER statistics between hypothesis and provided references.

    Args:
        pred_words: A list of tokenized hypothesis sentence.
        target_words: A list of lists of tokenized reference sentences.

    Return:
        best_num_edits:
            The best (lowest) number of required edits to match hypothesis and reference sentences.
        avg_tgt_len:
            Average length of tokenized reference sentences.
    """
    tgt_lengths = torch.tensor(0.0)
    best_num_edits = torch.tensor(2e16)

    for tgt_words in target_words:
        num_edits = _translation_edit_rate(tgt_words, pred_words)
        tgt_lengths += len(tgt_words)
        if num_edits < best_num_edits:
            best_num_edits = num_edits

    avg_tgt_len = tgt_lengths / len(target_words)
    return best_num_edits, avg_tgt_len


def _compute_ter_score_from_statistics(num_edits: torch.Tensor, tgt_length: torch.Tensor) -> torch.Tensor:
    """Compute TER score based on pre-computed a number of edits and an average reference length.

    Args:
        num_edits: A number of required edits to match hypothesis and reference sentences.
        tgt_length: An average length of reference sentences.

    Return:
        A corpus-level TER score or 1 if reference_length == 0.
    """
    if tgt_length > 0 and num_edits > 0:
        score = num_edits / tgt_length
    elif tgt_length == 0 and num_edits > 0:
        score = torch.tensor(1.0)
    else:
        score = torch.tensor(0.0)
    return score


def _remove_punct(sentence: str) -> str:
    """Remove punctuation from an input sentence string."""
    return re.sub(
        r"[\.,\?:;!\"\(\)"
        r"\u3001\u3002\u3008-\u3011\u3014-\u301f\uff61-\uff65\u30fb"
        r"\uff0e\uff0c\uff1f\uff1a\uff1b\uff01\uff02\uff08\uff09]", "", sentence
    )


class TER(tm.Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
            self,
            tokenize_fn: typing.Optional[typing.Callable] = None,
            no_punctuation: bool = False,
            lowercase: bool = True,
            return_sentence_level_score: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenize_fn = tokenize_fn
        if tokenize_fn is None:
            self.tokenize_fn = lambda x: x.split()
        self.no_punctuation = no_punctuation
        self.lowercase = lowercase
        self.return_sentence_level_score = return_sentence_level_score

        self.add_state("total_num_edits", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tgt_len", torch.tensor(0.0), dist_reduce_fx="sum")
        if self.return_sentence_level_score:
            self.add_state("sentence_ter", torch.tensor([]), dist_reduce_fx="cat")
        else:
            self.sentence_ter = None

    def update(
            self,
            preds: typing.Sequence[str],
            target: typing.Sequence[typing.Union[str, typing.Sequence[str]]]
    ):
        preds = [preds] if isinstance(preds, str) else preds
        target = [[tgt] if isinstance(tgt, str) else tgt for tgt in target]

        if self.lowercase:
            preds = [pred.lower() for pred in preds]
            target = [[tgt.lower() for tgt in tgt] for tgt in target]
        if self.no_punctuation:
            preds = [_remove_punct(pred) for pred in preds]
            target = [[_remove_punct(tgt) for tgt in tgt] for tgt in target]

        preds_words = [self.tokenize_fn(line) if line else [] for line in preds]
        target_words = [[self.tokenize_fn(line) for line in tgt] for tgt in target]

        for pred, tgt in zip(preds_words, target_words):
            num_edits, tgt_length = _compute_sentence_statistics(pred, tgt)
            self.total_num_edits += num_edits
            self.total_tgt_len += tgt_length
            if self.sentence_ter is not None:
                self.sentence_ter.append(_compute_ter_score_from_statistics(num_edits, tgt_length).unsqueeze(0))

    def compute(self):
        ter = _compute_ter_score_from_statistics(self.total_num_edits, self.total_tgt_len)
        if self.sentence_ter is not None:
            return ter, torch.cat(self.sentence_ter)
        return ter
