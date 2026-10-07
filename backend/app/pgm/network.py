"""
Probabilistic Graphical Models: Bayesian Network representation.
Loads DAG and CPTs from specification, validates acyclicity and normalization,
computes topological ordering, full joint probabilities, and performs ancestral sampling.

Syllabus Topic: Bayesian Networks, DAG structure, CPT representation, Generative sampling.
"""
from __future__ import annotations
import json
import itertools
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from backend.app.pgm.factor import Factor


class BayesNet:
    """
    Discrete Bayesian Network representation loaded from model_spec.json.
    """
    def __init__(self, spec: Dict[str, Any]):
        self.raw_spec = spec
        self.nodes_spec = spec["nodes"]
        self.decision_spec = spec.get("decision", {})
        self.vital_clinical_meaning = spec.get("vital_clinical_meaning", {})

        self.nodes: List[str] = list(self.nodes_spec.keys())
        self.states: Dict[str, List[str]] = {}
        self.cards: Dict[str, int] = {}
        self.parents: Dict[str, List[str]] = {}
        self.children: Dict[str, List[str]] = {n: [] for n in self.nodes}

        for n, d in self.nodes_spec.items():
            self.states[n] = list(d["states"])
            self.cards[n] = len(self.states[n])
            self.parents[n] = list(d.get("parents", []))

        for n, pars in self.parents.items():
            for p in pars:
                if p in self.children:
                    self.children[p].append(n)

        # Validate DAG and compute topological ordering
        self.topological_order = self._compute_topological_order()

        # Validate CPT row sums and build Factor objects
        self.cpts: Dict[str, Factor] = {}
        self._build_cpts()

    def _compute_topological_order(self) -> List[str]:
        """
        Kahn's algorithm for topological sorting and cycle detection.
        """
        in_degree = {n: len(self.parents[n]) for n in self.nodes}
        queue = [n for n in self.nodes if in_degree[n] == 0]
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for child in self.children[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            unvisited = [n for n in self.nodes if n not in order]
            raise ValueError(f"Graph contains a directed cycle! Unvisited nodes: {unvisited}")

        return order

    def _build_cpts(self):
        """
        Converts dictionary CPTs to Factor representations.
        Validates that every row sums to 1 (tol 1e-4).
        """
        for n, d in self.nodes_spec.items():
            pars = self.parents[n]
            n_states = self.states[n]
            n_card = self.cards[n]
            cpt_dict = d["cpt"]

            # Factor vars: parents in order, followed by n
            factor_vars = list(pars) + [n]
            factor_cards = [self.cards[p] for p in pars] + [n_card]

            # Allocate multi-dimensional array
            table = np.zeros(factor_cards, dtype=np.float64)

            if len(pars) == 0:
                row = cpt_dict.get("", cpt_dict.get("none", None))
                if row is None:
                    row = list(cpt_dict.values())[0]
                row_arr = np.asarray(row, dtype=np.float64)
                if abs(np.sum(row_arr) - 1.0) > 1e-4:
                    raise ValueError(f"CPT for root node {n} does not sum to 1: sum={np.sum(row_arr)}")
                table = row_arr
            else:
                # Iterate through all parent configurations in lexicographic index order
                dims = [self.cards[p] for p in pars]
                for idx in itertools.product(*[range(d) for d in dims]):
                    key = "|".join(self.states[p][i] for p, i in zip(pars, idx))
                    if key not in cpt_dict:
                        raise KeyError(f"Missing CPT entry '{key}' for node '{n}'")
                    row = cpt_dict[key]
                    row_arr = np.asarray(row, dtype=np.float64)
                    if abs(np.sum(row_arr) - 1.0) > 1e-4:
                        raise ValueError(f"CPT row '{key}' for node '{n}' does not sum to 1: sum={np.sum(row_arr)}")
                    table[idx] = row_arr

            self.cpts[n] = Factor(factor_vars, factor_cards, table)

    def joint_probability(self, assignment: Dict[str, str | int]) -> float:
        """
        Computes P(X1=x1, ..., Xn=xn) by the chain rule:
        P(X) = prod_{i=1}^n P(X_i | Parents(X_i))
        """
        prob = 1.0
        for n in self.topological_order:
            val = assignment[n]
            val_idx = val if isinstance(val, int) else self.states[n].index(val)
            
            pars = self.parents[n]
            if len(pars) == 0:
                key = ""
                prob *= self.nodes_spec[n]["cpt"][key][val_idx]
            else:
                p_states = []
                for p in pars:
                    p_val = assignment[p]
                    p_str = self.states[p][p_val] if isinstance(p_val, int) else p_val
                    p_states.append(p_str)
                key = "|".join(p_states)
                prob *= self.nodes_spec[n]["cpt"][key][val_idx]
        return float(prob)

    def ancestral_sample(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> List[Dict[str, str]]:
        """
        Forward ancestral sampling in topological order.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        samples = []
        for _ in range(n_samples):
            assignment: Dict[str, str] = {}
            for n in self.topological_order:
                pars = self.parents[n]
                if len(pars) == 0:
                    probs = self.nodes_spec[n]["cpt"][""]
                else:
                    key = "|".join(assignment[p] for p in pars)
                    probs = self.nodes_spec[n]["cpt"][key]
                chosen_idx = rng.choice(self.cards[n], p=probs)
                assignment[n] = self.states[n][chosen_idx]
            samples.append(assignment)
        return samples

    def do_intervene(self, interventions: Dict[str, str]) -> BayesNet:
        """
        Pearl's do-calculus: graph surgery for do(X = x).
        Sever all incoming edges to X, and set P(X=x) = 1.0.
        Returns a new BayesNet instance representing the mutilated graph.
        """
        new_spec = json.loads(json.dumps(self.raw_spec))
        for target_var, target_val in interventions.items():
            if target_var not in new_spec["nodes"]:
                continue
            states = new_spec["nodes"][target_var]["states"]
            if target_val not in states:
                raise ValueError(f"State '{target_val}' not valid for variable '{target_var}'")
            target_idx = states.index(target_val)
            # Remove parents
            new_spec["nodes"][target_var]["parents"] = []
            # Make deterministic
            prob_vec = [1.0 if i == target_idx else 0.0 for i in range(len(states))]
            new_spec["nodes"][target_var]["cpt"] = {"": prob_vec}
        return BayesNet(new_spec)

    @classmethod
    def from_file(cls, path: str) -> BayesNet:
        with open(path, "r") as f:
            spec = json.load(f)
        return cls(spec)
