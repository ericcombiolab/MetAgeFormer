"""Fast Harrell C-index for right-censored survival (O(n log n), numpy-only)."""

from __future__ import annotations

import numpy as np


class _BTree:
    """Balanced order-statistic tree (lifelines-style) for concordance counting."""

    def __init__(self, values: np.ndarray):
        self._tree = self._treeify(values)
        self._counts = np.zeros_like(self._tree, dtype=np.int64)

    @staticmethod
    def _treeify(values: np.ndarray) -> np.ndarray:
        if len(values) == 1:
            return values
        tree = np.empty_like(values)
        last_full_row = int(np.log2(len(values) + 1) - 1)
        len_ragged_row = len(values) - (2 ** (last_full_row + 1) - 1)
        if len_ragged_row > 0:
            bottom_row_ix = np.s_[: 2 * len_ragged_row : 2]
            tree[-len_ragged_row:] = values[bottom_row_ix]
            values = np.delete(values, bottom_row_ix)
        values_start = 0
        values_space = 2
        values_len = 2 ** last_full_row
        while values_start < len(values):
            tree[values_len - 1 : 2 * values_len - 1] = values[values_start::values_space]
            values_start += int(values_space / 2)
            values_space *= 2
            values_len = int(values_len / 2)
        return tree

    def insert(self, value: float) -> None:
        i = 0
        n = len(self._tree)
        while i < n:
            cur = self._tree[i]
            self._counts[i] += 1
            if value < cur:
                i = 2 * i + 1
            elif value > cur:
                i = 2 * i + 2
            else:
                return
        raise ValueError(f"Value {value} not contained in tree.")

    def __len__(self) -> int:
        return int(self._counts[0])

    def rank(self, value: float) -> tuple[int, int]:
        i = 0
        n = len(self._tree)
        rank = 0
        count = 0
        while i < n:
            cur = self._tree[i]
            if value < cur:
                i = 2 * i + 1
                continue
            if value > cur:
                rank += self._counts[i]
                nexti = 2 * i + 2
                if nexti < n:
                    rank -= self._counts[nexti]
                i = nexti
                continue
            count = self._counts[i]
            lefti = 2 * i + 1
            if lefti < n:
                nleft = self._counts[lefti]
                count -= nleft
                rank += nleft
                righti = lefti + 1
                if righti < n:
                    count -= self._counts[righti]
            return rank, count
        return rank, count


def _handle_pairs(truth, pred, first_ix, times_to_compare):
    next_ix = first_ix
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = len(times_to_compare) * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
        rank, count = times_to_compare.rank(pred[i])
        correct += rank
        tied += count
    return pairs, correct, tied, next_ix


def fast_cindex(risk: np.ndarray, event: np.ndarray, time: np.ndarray) -> float:
    """Harrell's C-index for Cox-style risks (higher risk = worse prognosis)."""
    risk = -np.asarray(risk, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=bool).ravel()
    time = np.asarray(time, dtype=np.float64).ravel()
    if risk.shape != time.shape or risk.shape != event.shape:
        raise ValueError("risk, event, and time must have the same shape")
    if risk.size == 0:
        return float("nan")

    if not event.any():
        return float("nan")

    died_mask = event
    died_truth = time[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]
    died_pred = risk[died_mask][ix]

    censored_truth = time[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = risk[~died_mask][ix]

    unique_preds = np.unique(risk)
    if unique_preds.size == 0:
        return float("nan")
    times_to_compare = _BTree(unique_preds)

    censored_ix = 0
    died_ix = 0
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        if has_more_censored and (not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, censored_ix = _handle_pairs(
                censored_truth, censored_pred, censored_ix, times_to_compare
            )
        elif has_more_died and (not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = _handle_pairs(
                died_truth, died_pred, died_ix, times_to_compare
            )
            for pred in died_pred[died_ix:next_ix]:
                times_to_compare.insert(pred)
            died_ix = next_ix
        else:
            break
        num_pairs += pairs
        num_correct += correct
        num_tied += tied

    if num_pairs == 0:
        return float("nan")
    return float((num_correct + num_tied / 2) / num_pairs)
