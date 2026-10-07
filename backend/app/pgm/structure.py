"""
Probabilistic Graphical Models: Structure Learning & Hill Climbing with BIC Score.
Implements:
  - Decomposable Bayesian Information Criterion (BIC) score:
    Score(G : D) = sum_i [ LL(X_i | Pa_i) - 0.5 * dim(X_i | Pa_i) * log(N) ]
  - Hill climbing search with Add, Delete, Reverse edge operators
  - Acyclicity maintenance (Cycle detection)
  - Structural Hamming Distance (SHD) vs Ground-Truth DAG
  - Edge comparison matrix (correct, missing, extra, reversed)

Syllabus Topic: Structure Learning, Score-based Search, BIC, Model Selection.
"""
from __future__ import annotations
import copy
from typing import Dict, List, Set, Tuple, Any, Optional
import itertools
import numpy as np
import pandas as pd
from backend.app.pgm.network import BayesNet


class StructureLearner:
    def __init__(self, net: BayesNet):
        self.net = net

    def local_bic_score(
        self,
        node: str,
        parents: List[str] | Set[str],
        df: pd.DataFrame
    ) -> float:
        """
        Computes local BIC score for node given parents:
        BIC(X | Pa) = sum_{pa, x} Count(x, pa) * log(Count(x, pa) / Count(pa)) - 0.5 * dim * log(N)
        where dim = (card(X) - 1) * prod_{P in Pa} card(P)
        """
        pars = list(parents)
        states = self.net.states[node]
        r_i = len(states)
        q_i = int(np.prod([self.net.cards[p] for p in pars])) if len(pars) > 0 else 1
        dim = (r_i - 1) * q_i
        N = len(df)

        if len(pars) == 0:
            counts = df[node].value_counts().values.astype(np.float64)
            probs = counts / counts.sum()
            valid = (counts > 0) & (probs > 0)
            ll = np.sum(counts[valid] * np.log(probs[valid]))
        else:
            # Grouped counts
            grouped = df.groupby(pars + [node]).size()
            par_grouped = df.groupby(pars).size()

            ll = 0.0
            for idx, count in grouped.items():
                par_idx = idx[:-1] if len(pars) > 1 else idx[0]
                total_par = par_grouped.get(par_idx, 0)
                if total_par > 0 and count > 0:
                    ll += count * np.log(count / total_par)

        penalty = 0.5 * dim * np.log(max(1, N))
        return float(ll - penalty)

    def total_bic_score(
        self,
        dag_parents: Dict[str, List[str]],
        df: pd.DataFrame
    ) -> float:
        """
        Total BIC score: sum over all nodes of their local score.
        """
        return sum(self.local_bic_score(n, dag_parents[n], df) for n in dag_parents)

    def is_acyclic(self, dag_parents: Dict[str, List[str]], nodes: List[str]) -> bool:
        """
        Kahn's cycle detection.
        """
        in_degree = {n: len(dag_parents[n]) for n in nodes}
        children: Dict[str, List[str]] = {n: [] for n in nodes}
        for n, pars in dag_parents.items():
            for p in pars:
                children[p].append(n)

        queue = [n for n in nodes if in_degree[n] == 0]
        visited = 0
        while queue:
            curr = queue.pop(0)
            visited += 1
            for ch in children[curr]:
                in_degree[ch] -= 1
                if in_degree[ch] == 0:
                    queue.append(ch)

        return visited == len(nodes)

    def hill_climbing(
        self,
        df: pd.DataFrame,
        max_parents: int = 3,
        max_iterations: int = 40,
        tabu_size: int = 5,
        target_nodes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Greedy hill-climbing search with score cache, tabu list, and acyclicity constraint.
        """
        nodes = target_nodes or [
            "Age", "History", "Condition", "Severity", "ChestPain", "Dyspnea",
            "HeartRate", "SystolicBP", "SpO2", "ArrivalMode"
        ]
        # Initial empty graph or starting DAG
        current_parents: Dict[str, List[str]] = {n: [] for n in nodes}

        # Cache local scores
        local_scores = {n: self.local_bic_score(n, [], df) for n in nodes}
        current_score = sum(local_scores.values())

        score_history = [round(current_score, 2)]
        tabu_list: List[Dict[str, List[str]]] = []

        for step in range(max_iterations):
            best_move = None
            best_score_delta = float("-inf")
            best_new_local = None
            best_affected_nodes = None

            # Explore neighborhood moves: Add, Delete, Reverse
            for u in nodes:
                for v in nodes:
                    if u == v:
                        continue

                    # Move 1: Add edge u -> v
                    if u not in current_parents[v] and len(current_parents[v]) < max_parents:
                        cand_parents = {n: list(current_parents[n]) for n in nodes}
                        cand_parents[v].append(u)
                        if self.is_acyclic(cand_parents, nodes):
                            cand_local_v = self.local_bic_score(v, cand_parents[v], df)
                            delta = cand_local_v - local_scores[v]
                            if delta > best_score_delta:
                                best_score_delta = delta
                                best_move = ("ADD", u, v)
                                best_new_local = {v: cand_local_v}
                                best_affected_nodes = [v]

                    # Move 2: Delete edge u -> v
                    if u in current_parents[v]:
                        cand_parents = {n: list(current_parents[n]) for n in nodes}
                        cand_parents[v].remove(u)
                        cand_local_v = self.local_bic_score(v, cand_parents[v], df)
                        delta = cand_local_v - local_scores[v]
                        if delta > best_score_delta:
                            best_score_delta = delta
                            best_move = ("DELETE", u, v)
                            best_new_local = {v: cand_local_v}
                            best_affected_nodes = [v]

                    # Move 3: Reverse edge u -> v to v -> u
                    if u in current_parents[v] and len(current_parents[u]) < max_parents:
                        cand_parents = {n: list(current_parents[n]) for n in nodes}
                        cand_parents[v].remove(u)
                        cand_parents[u].append(v)
                        if self.is_acyclic(cand_parents, nodes):
                            cand_local_v = self.local_bic_score(v, cand_parents[v], df)
                            cand_local_u = self.local_bic_score(u, cand_parents[u], df)
                            delta = (cand_local_v - local_scores[v]) + (cand_local_u - local_scores[u])
                            if delta > best_score_delta:
                                best_score_delta = delta
                                best_move = ("REVERSE", u, v)
                                best_new_local = {v: cand_local_v, u: cand_local_u}
                                best_affected_nodes = [v, u]

            # If no move improves score, terminate
            if best_move is None or best_score_delta <= 1e-4:
                break

            # Apply best move
            op, u, v = best_move
            if op == "ADD":
                current_parents[v].append(u)
            elif op == "DELETE":
                current_parents[v].remove(u)
            elif op == "REVERSE":
                current_parents[v].remove(u)
                current_parents[u].append(v)

            # Update cached scores
            for n, sc in best_new_local.items():
                local_scores[n] = sc
            current_score += best_score_delta
            score_history.append(round(current_score, 2))

        # Evaluate Structural Hamming Distance (SHD) against true graph
        shd_metrics = self.compute_shd(current_parents, nodes)

        return {
            "target_nodes": nodes,
            "learned_edges": [
                {"source": p, "target": n} for n in nodes for p in current_parents[n]
            ],
            "score_history": score_history,
            "final_bic_score": round(current_score, 2),
            "shd_metrics": shd_metrics
        }

    def compute_shd(
        self,
        learned_parents: Dict[str, List[str]],
        active_nodes: List[str]
    ) -> Dict[str, Any]:
        """
        Computes Structural Hamming Distance (SHD), true edges, extra edges, missing edges.
        """
        true_edges = set()
        for n in active_nodes:
            for p in self.net.parents.get(n, []):
                if p in active_nodes:
                    true_edges.add((p, n))

        learned_edges = set()
        for n in active_nodes:
            for p in learned_parents.get(n, []):
                learned_edges.add((p, n))

        correct = true_edges & learned_edges
        missing = true_edges - learned_edges
        extra = learned_edges - true_edges

        # Check reversed edges
        reversed_edges = set()
        for (u, v) in list(missing):
            if (v, u) in extra:
                reversed_edges.add((u, v))
                missing.remove((u, v))
                extra.remove((v, u))

        shd = len(missing) + len(extra) + len(reversed_edges)

        return {
            "shd": shd,
            "total_true_edges": len(true_edges),
            "correct_edges": [{"source": u, "target": v} for u, v in correct],
            "missing_edges": [{"source": u, "target": v} for u, v in missing],
            "extra_edges": [{"source": u, "target": v} for u, v in extra],
            "reversed_edges": [{"source": u, "target": v} for u, v in reversed_edges]
        }
