"""
Probabilistic Graphical Models: d-Separation & I-Maps.
Implements:
  - Bayes-Ball / Active Trail reachability algorithm for d-separation check:
    X _|_ Y | Z
  - Explicit active and blocked trail analysis (chain, fork, collider status)
  - Empirical Mutual Information calculation from synthetic dataset
  - Verification of standard graphical independence relations

Syllabus Topic: Conditional Independence, d-Separation, I-Maps, Bayes-Ball.
"""
from __future__ import annotations
from typing import Dict, List, Set, Any, Tuple, Optional
import itertools
import numpy as np
import pandas as pd
from backend.app.pgm.network import BayesNet


class DSeparationAnalyzer:
    def __init__(self, net: BayesNet):
        self.net = net

    def _get_ancestors(self, nodes: Set[str]) -> Set[str]:
        """
        Finds all ancestors of a set of nodes in the DAG.
        """
        ancestors = set(nodes)
        queue = list(nodes)
        while queue:
            curr = queue.pop(0)
            for p in self.net.parents[curr]:
                if p not in ancestors:
                    ancestors.add(p)
                    queue.append(p)
        return ancestors

    def is_dseparated(
        self,
        X: str,
        Y: str,
        Z: List[str] | Set[str]
    ) -> Dict[str, Any]:
        """
        Tests if X is d-separated from Y given conditioning set Z.
        Also traces all simple undirected paths and identifies why each is active or blocked.
        """
        Z_set = set(Z)
        if X == Y or X in Z_set or Y in Z_set:
            return {
                "independent": False,
                "x": X,
                "y": Y,
                "z": list(Z_set),
                "active_paths": [{"path": [X, Y], "reason": "Trivial overlap in query sets"}],
                "blocked_paths": []
            }

        # Precompute ancestors of Z for collider reachability
        Z_ancestors = self._get_ancestors(Z_set)

        # Find all simple undirected paths between X and Y (length <= 7 to avoid combinatorial blowup)
        # Undirected neighbors
        neighbors: Dict[str, Set[str]] = {n: set(self.net.parents[n]) | set(self.net.children[n]) for n in self.net.nodes}

        all_paths: List[List[str]] = []

        def find_paths(current: str, target: str, visited: List[str]):
            if len(visited) > 7:
                return
            if current == target:
                all_paths.append(list(visited))
                return
            for nbr in neighbors[current]:
                if nbr not in visited:
                    visited.append(nbr)
                    find_paths(nbr, target, visited)
                    visited.pop()

        find_paths(X, Y, [X])

        active_paths = []
        blocked_paths = []

        for path in all_paths:
            # Check if path is active
            path_is_active = True
            blocking_reason = None

            # Examine every interior node path[i] with predecessor path[i-1] and successor path[i+1]
            for i in range(1, len(path) - 1):
                prev = path[i - 1]
                node = path[i]
                nxt = path[i + 1]

                # Determine trail structure at node:
                # Collider: prev -> node <- nxt
                is_collider = (prev in self.net.parents[node]) and (nxt in self.net.parents[node])

                if is_collider:
                    # Collider is active iff node in Z or node in ancestors of Z
                    if node not in Z_ancestors:
                        path_is_active = False
                        blocking_reason = f"Collider at '{node}' ({prev} -> {node} <- {nxt}) is NOT observed nor has observed descendants."
                        break
                else:
                    # Non-collider (chain or fork)
                    # Chain: prev -> node -> nxt OR prev <- node <- nxt
                    # Fork: prev <- node -> nxt
                    # Active iff node is NOT in Z
                    if node in Z_set:
                        path_is_active = False
                        struct_type = "Fork" if (prev in self.net.children[node] and nxt in self.net.children[node]) else "Chain"
                        blocking_reason = f"{struct_type} at '{node}' is OBSERVED in conditioning set Z."
                        break

            path_str = " - ".join(path)
            if path_is_active:
                active_paths.append({
                    "path": path,
                    "trail": path_str,
                    "reason": "All colliders activated and all non-colliders unobserved."
                })
            else:
                blocked_paths.append({
                    "path": path,
                    "trail": path_str,
                    "reason": blocking_reason
                })

        is_independent = (len(active_paths) == 0)

        explanation = (
            f"'{X}' and '{Y}' are conditionally INDEPENDENT given {{{', '.join(Z_set)}}}. "
            f"All {len(blocked_paths)} path(s) are blocked by d-separation."
            if is_independent else
            f"'{X}' and '{Y}' are conditionally DEPENDENT given {{{', '.join(Z_set)}}}. "
            f"There are {len(active_paths)} active trail(s) transmitting probabilistic influence."
        )

        return {
            "x": X,
            "y": Y,
            "z": sorted(list(Z_set)),
            "independent": is_independent,
            "d_separated": is_independent,
            "active_paths_count": len(active_paths),
            "blocked_paths_count": len(blocked_paths),
            "active_paths": active_paths,
            "blocked_paths": blocked_paths,
            "explanation": explanation
        }

    def compute_empirical_mutual_info(
        self,
        df: pd.DataFrame,
        X: str,
        Y: str,
        Z: Optional[List[str]] = None
    ) -> float:
        """
        Computes conditional mutual information I(X; Y | Z) from data:
        I(X; Y | Z) = sum_{z} P(z) * sum_{x, y} P(x, y | z) * log( P(x, y | z) / (P(x|z) * P(y|z)) )
        When Z is empty, computes standard Mutual Information I(X; Y).
        """
        if Z is None or len(Z) == 0:
            # Unconditional MI
            counts_xy = pd.crosstab(df[X], df[Y])
            p_xy = (counts_xy / len(df)).values
            p_x = p_xy.sum(axis=1, keepdims=True)
            p_y = p_xy.sum(axis=0, keepdims=True)

            prod = p_x @ p_y
            valid = (p_xy > 1e-12) & (prod > 1e-12)
            mi = np.sum(p_xy[valid] * np.log(p_xy[valid] / prod[valid]))
            return float(max(0.0, mi))

        # Conditional MI: group by Z
        mi_cond = 0.0
        n_total = len(df)

        grouped = df.groupby(Z)
        for _, group in grouped:
            weight = len(group) / n_total
            if len(group) < 5:
                continue
            counts_xy = pd.crosstab(group[X], group[Y])
            p_xy = (counts_xy / len(group)).values
            p_x = p_xy.sum(axis=1, keepdims=True)
            p_y = p_xy.sum(axis=0, keepdims=True)
            prod = p_x @ p_y
            valid = (p_xy > 1e-12) & (prod > 1e-12)
            sub_mi = np.sum(p_xy[valid] * np.log(p_xy[valid] / prod[valid]))
            mi_cond += weight * max(0.0, sub_mi)

        return float(mi_cond)
