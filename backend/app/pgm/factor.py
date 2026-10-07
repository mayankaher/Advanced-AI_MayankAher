"""
Probabilistic Graphical Models: Discrete Factor Engine.
Implements Factor algebra (multiplication, marginalization, reduction, normalization, maximization)
using pure NumPy broadcasting.

Syllabus Topic: Probability theory, Factor representations, Exact Inference primitives.
Complexity:
  - Multiplication: O(prod(cards(vars_union)))
  - Marginalization / Maximization: O(prod(cards(self.vars)))
  - Reduction: O(prod(cards(remaining_vars)))
"""
from __future__ import annotations
from typing import List, Sequence, Dict, Any, Tuple, Optional
import numpy as np


class Factor:
    """
    A multi-dimensional discrete potential/factor over a set of random variables.
    """
    def __init__(
        self,
        vars: Sequence[str],
        cards: Sequence[int],
        values: np.ndarray | Sequence[float]
    ):
        self.vars: List[str] = list(vars)
        self.cards: List[int] = [int(c) for c in cards]
        val_arr = np.asarray(values, dtype=np.float64)
        expected_size = int(np.prod(self.cards)) if len(self.cards) > 0 else 1
        if val_arr.size != expected_size:
            raise ValueError(
                f"Size of values ({val_arr.size}) does not match product of cardinalities ({self.cards})"
            )
        self.values: np.ndarray = val_arr.reshape(self.cards) if len(self.cards) > 0 else val_arr.reshape(())


    def copy(self) -> Factor:
        return Factor(list(self.vars), list(self.cards), self.values.copy())

    def multiply(self, other: Factor) -> Factor:
        """
        Factor multiplication (phi1 * phi2) using NumPy broadcasting.
        Scope of result is the union of variables.
        """
        if len(self.vars) == 0:
            return Factor(other.vars, other.cards, self.values * other.values)
        if len(other.vars) == 0:
            return Factor(self.vars, self.cards, self.values * other.values)

        # Union of variables maintaining stable ordering
        new_vars: List[str] = list(self.vars)
        for v in other.vars:
            if v not in new_vars:
                new_vars.append(v)

        # Map current factor axes to new_vars
        card_map: Dict[str, int] = {}
        for v, c in zip(self.vars, self.cards):
            card_map[v] = c
        for v, c in zip(other.vars, other.cards):
            card_map[v] = c
        new_cards = [card_map[v] for v in new_vars]

        # Reshape self.values to align with new_vars
        self_shape = [1] * len(new_vars)
        for i, v in enumerate(self.vars):
            new_idx = new_vars.index(v)
            self_shape[new_idx] = self.cards[i]
        # Transpose self into order matching new_vars
        self_axes = [new_vars.index(v) for v in self.vars]
        # Reorder self.values so axes match their appearance in new_vars
        # Alternative: use transpose to align axes
        v1 = self.values
        # Create an expanded view of v1 with length len(new_vars)
        slice_shape1 = [1] * len(new_vars)
        for old_axis, v in enumerate(self.vars):
            new_axis = new_vars.index(v)
            slice_shape1[new_axis] = self.cards[old_axis]
        
        # We need to reshape v1 correctly by first transposing if necessary,
        # but easier: v1 has shape self.cards.
        # We can construct the view by inserting newaxes where variables are absent.
        # To make shape match slice_shape1, we need to order axes of v1 according to new_vars:
        perm1 = np.argsort(self_axes) # order that sorts self.vars to match new_vars order
        # Actually: if we transpose v1 by argsort(self_axes), its axes are in sorted order of new_vars indices.
        # Then we reshape by inserting 1s at positions not in self_axes.
        v1_transposed = np.transpose(v1, perm1) if len(self.vars) > 1 else v1
        # Now expand v1_transposed with 1s:
        expanded_shape1 = []
        curr = 0
        sorted_axes = sorted(self_axes)
        for axis_idx in range(len(new_vars)):
            if axis_idx in sorted_axes:
                expanded_shape1.append(new_cards[axis_idx])
                curr += 1
            else:
                expanded_shape1.append(1)
        v1_expanded = v1_transposed.reshape(expanded_shape1)

        # Do the same for other
        other_axes = [new_vars.index(v) for v in other.vars]
        perm2 = np.argsort(other_axes)
        v2_transposed = np.transpose(other.values, perm2) if len(other.vars) > 1 else other.values
        sorted_axes2 = sorted(other_axes)
        expanded_shape2 = []
        for axis_idx in range(len(new_vars)):
            if axis_idx in sorted_axes2:
                expanded_shape2.append(new_cards[axis_idx])
            else:
                expanded_shape2.append(1)
        v2_expanded = v2_transposed.reshape(expanded_shape2)

        new_values = v1_expanded * v2_expanded
        return Factor(new_vars, new_cards, new_values)

    def marginalize(self, vars_to_sum: Sequence[str] | str) -> Factor:
        """
        Sum-out the specified variable(s).
        """
        if isinstance(vars_to_sum, str):
            vars_to_sum = [vars_to_sum]
        
        axes_to_sum = []
        new_vars = []
        new_cards = []
        for i, v in enumerate(self.vars):
            if v in vars_to_sum:
                axes_to_sum.append(i)
            else:
                new_vars.append(v)
                new_cards.append(self.cards[i])

        if not axes_to_sum:
            return self.copy()

        new_values = np.sum(self.values, axis=tuple(axes_to_sum))
        return Factor(new_vars, new_cards, new_values)

    def maximize(self, vars_to_max: Sequence[str] | str) -> Tuple[Factor, Optional[np.ndarray]]:
        """
        Max-out the specified variable(s) (for MAP inference).
        Returns (max_factor, argmax_indices).
        """
        if isinstance(vars_to_max, str):
            vars_to_max = [vars_to_max]

        axes_to_max = []
        new_vars = []
        new_cards = []
        for i, v in enumerate(self.vars):
            if v in vars_to_max:
                axes_to_max.append(i)
            else:
                new_vars.append(v)
                new_cards.append(self.cards[i])

        if not axes_to_max:
            return self.copy(), None

        # For single variable max-out:
        if len(axes_to_max) == 1:
            axis = axes_to_max[0]
            max_values = np.max(self.values, axis=axis)
            argmax_indices = np.argmax(self.values, axis=axis)
            return Factor(new_vars, new_cards, max_values), argmax_indices

        # Multiple axes: flatten axes or iterate
        res = self.values
        # To keep track of argmax for multiple, sequentially max-out
        cur_f = self.copy()
        for v in vars_to_max:
            cur_f, _ = cur_f.maximize(v)
        return cur_f, None

    def reduce(self, evidence: Dict[str, int]) -> Factor:
        """
        Reduce the factor by conditioning on observed variables.
        evidence is a mapping from var_name -> state_index.
        Observed variables are sliced out from the factor scope.
        """
        relevant_ev = {v: evidence[v] for v in self.vars if v in evidence}
        if not relevant_ev:
            return self.copy()

        slices = []
        new_vars = []
        new_cards = []
        for i, v in enumerate(self.vars):
            if v in relevant_ev:
                idx = relevant_ev[v]
                if idx < 0 or idx >= self.cards[i]:
                    raise IndexError(f"Evidence index {idx} out of range for variable {v} (card {self.cards[i]})")
                slices.append(idx)
            else:
                slices.append(slice(None))
                new_vars.append(v)
                new_cards.append(self.cards[i])

        reduced_vals = self.values[tuple(slices)]
        return Factor(new_vars, new_cards, reduced_vals)

    def normalize(self) -> Factor:
        """
        Normalize values so they sum to 1.
        """
        total = float(np.sum(self.values))
        if total == 0 or np.isnan(total):
            # Uniform fallback
            if self.values.size > 0:
                new_vals = np.ones_like(self.values, dtype=np.float64) / self.values.size
            else:
                new_vals = self.values.copy()
        else:
            new_vals = self.values / total
        return Factor(self.vars, self.cards, new_vals)

    def get_distribution(self, state_names: Optional[Dict[str, List[str]]] = None) -> Dict[str, float]:
        """
        Returns a dict mapping state name (or assignment string) to probability.
        Assumes factor is 1D (single target variable) or returns flattened mapping.
        """
        norm_f = self.normalize()
        if len(self.vars) == 1:
            var = self.vars[0]
            vals = norm_f.values.flatten()
            if state_names and var in state_names:
                return {state_names[var][i]: float(vals[i]) for i in range(len(vals))}
            return {str(i): float(vals[i]) for i in range(len(vals))}
        
        # Multidimensional: enumerate assignments
        res = {}
        for idx in np.ndindex(tuple(self.cards)):
            parts = []
            for d, var in enumerate(self.vars):
                s_name = state_names[var][idx[d]] if state_names and var in state_names else str(idx[d])
                parts.append(f"{var}={s_name}")
            res[", ".join(parts)] = float(norm_f.values[idx])
        return res

    def __repr__(self) -> str:
        return f"Factor(vars={self.vars}, cards={self.cards}, total_sum={float(np.sum(self.values)):.4f})"
