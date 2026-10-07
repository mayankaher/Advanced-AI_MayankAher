"""
Probabilistic Graphical Models: Exact Inference by Variable Elimination.
Implements:
  - Barren node pruning (unobserved leaves not in query targets)
  - Elimination order heuristics: min-neighbors (min-degree), min-fill, custom
  - Factor reduction by evidence
  - Step-by-step elimination trace generation (for pedagogical UI visualization)
  - Query method returning calibrated posterior Factor and trace

Syllabus Topic: Exact Inference, Variable Elimination, Factor Operations, Graph Heuristics.
Complexity:
  - Time: O(m * d^(w* + 1)) where w* is induced tree-width, d is max cardinality, m is number of factors.
  - Space: O(m * d^(w*))
"""
from __future__ import annotations
from typing import Dict, List, Set, Sequence, Any, Optional, Tuple
import numpy as np
from backend.app.pgm.factor import Factor
from backend.app.pgm.network import BayesNet


class EliminationTraceStep:
    def __init__(
        self,
        step: int,
        eliminated_var: str,
        input_factors_scopes: List[List[str]],
        product_scope: List[str],
        result_scope: List[str],
        result_size: int,
        description: str
    ):
        self.step = step
        self.eliminated_var = eliminated_var
        self.input_factors_scopes = input_factors_scopes
        self.product_scope = product_scope
        self.result_scope = result_scope
        self.result_size = result_size
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "eliminated_var": self.eliminated_var,
            "input_factors_scopes": self.input_factors_scopes,
            "product_scope": self.product_scope,
            "result_scope": self.result_scope,
            "result_size": self.result_size,
            "description": self.description
        }


class VariableElimination:
    def __init__(self, net: BayesNet):
        self.net = net

    def prune_barren_nodes(
        self,
        targets: Sequence[str],
        evidence: Dict[str, Any]
    ) -> Set[str]:
        """
        Iteratively removes unobserved leaf nodes not in targets or evidence.
        Why valid: In a DAG, summing out an unobserved leaf Y with parents Pa(Y)
        computes sum_y P(y | Pa(Y)) = 1.0. This factor cancels out to unity,
        leaving no effect on the distribution of any remaining variables.
        """
        targets_set = set(targets)
        evidence_set = set(evidence.keys())
        active_nodes = set(self.net.nodes)

        changed = True
        while changed:
            changed = False
            # Find nodes that have no active children and are neither targets nor evidence
            for n in list(active_nodes):
                if n in targets_set or n in evidence_set:
                    continue
                # Check if all children in net are inactive
                active_children = [c for c in self.net.children[n] if c in active_nodes]
                if len(active_children) == 0:
                    active_nodes.remove(n)
                    changed = True

        return active_nodes

    def _build_interaction_graph(
        self,
        factors: List[Factor]
    ) -> Dict[str, Set[str]]:
        """
        Builds the undirected interaction graph where an edge exists between two
        variables if they appear together in the scope of any factor.
        """
        adj: Dict[str, Set[str]] = {}
        for f in factors:
            for v in f.vars:
                if v not in adj:
                    adj[v] = set()
            for v1 in f.vars:
                for v2 in f.vars:
                    if v1 != v2:
                        adj[v1].add(v2)
        return adj

    def choose_order(
        self,
        vars_to_eliminate: List[str],
        factors: List[Factor],
        heuristic: str = "min-fill"
    ) -> List[str]:
        """
        Greedy ordering heuristics:
          - 'min-neighbors' (min degree in interaction graph)
          - 'min-fill' (min number of edges to add to make neighbors a clique)
        """
        remaining = set(vars_to_eliminate)
        adj = self._build_interaction_graph(factors)
        for v in vars_to_eliminate:
            if v not in adj:
                adj[v] = set()

        order: List[str] = []

        while remaining:
            best_var = None
            best_score = float("inf")

            for v in remaining:
                neighbors = adj[v] & remaining
                if heuristic == "min-neighbors":
                    score = len(neighbors)
                elif heuristic == "min-fill":
                    # Count missing edges between pairs of neighbors
                    fill_edges = 0
                    nbr_list = list(neighbors)
                    for i in range(len(nbr_list)):
                        for j in range(i + 1, len(nbr_list)):
                            if nbr_list[j] not in adj.get(nbr_list[i], set()):
                                fill_edges += 1
                    score = fill_edges
                else:
                    score = 0

                if score < best_score:
                    best_score = score
                    best_var = v

            assert best_var is not None
            order.append(best_var)
            remaining.remove(best_var)

            # Eliminate best_var: connect all neighbors to form a clique
            nbrs = list(adj[best_var])
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    u1, u2 = nbrs[i], nbrs[j]
                    adj[u1].add(u2)
                    adj[u2].add(u1)
            # Remove best_var from interaction graph
            del adj[best_var]
            for nbr in nbrs:
                adj[nbr].discard(best_var)

        return order

    def query(
        self,
        targets: Sequence[str],
        evidence: Optional[Dict[str, str]] = None,
        elimination_order: Optional[List[str]] = None,
        heuristic: str = "min-fill",
        prune_barren: bool = True
    ) -> Tuple[Factor, List[Dict[str, Any]]]:
        """
        Executes Variable Elimination:
          1. Prunes barren nodes
          2. Reduces factors by evidence
          3. Eliminates intermediate variables in order
          4. Multiplies remaining factors and normalizes over targets
        Returns (posterior_factor, trace_steps).
        """
        if evidence is None:
            evidence = {}

        # Convert string evidence to indices
        evidence_indices: Dict[str, int] = {}
        for var, val in evidence.items():
            if var not in self.net.nodes:
                raise ValueError(f"Unknown variable in evidence: {var}")
            if val not in self.net.states[var]:
                raise ValueError(f"Unknown state '{val}' for variable '{var}' (valid: {self.net.states[var]})")
            evidence_indices[var] = self.net.states[var].index(val)

        # 1. Prune barren nodes
        if prune_barren:
            active_nodes = self.prune_barren_nodes(targets, evidence)
        else:
            active_nodes = set(self.net.nodes)

        # 2. Collect initial CPT factors for active nodes and reduce by evidence
        active_factors: List[Factor] = []
        for n in active_nodes:
            f = self.net.cpts[n]
            # Reduce factor by evidence
            f_reduced = f.reduce(evidence_indices)
            if f_reduced.values.size > 0:
                active_factors.append(f_reduced)

        # Variables that need to be eliminated:
        # All active variables that are NOT in targets and NOT in evidence
        vars_to_elim = [
            v for v in active_nodes
            if v not in targets and v not in evidence_indices
        ]

        if elimination_order is None:
            elim_order = self.choose_order(vars_to_elim, active_factors, heuristic=heuristic)
        else:
            # Filter user order to only relevant variables
            elim_order = [v for v in elimination_order if v in vars_to_elim]
            # Add any missing
            for v in vars_to_elim:
                if v not in elim_order:
                    elim_order.append(v)

        trace: List[Dict[str, Any]] = []
        current_factors = list(active_factors)

        # 3. Eliminate each variable
        for step_idx, var in enumerate(elim_order, 1):
            # Partition factors into those mentioning var and those that do not
            relevant = [f for f in current_factors if var in f.vars]
            non_relevant = [f for f in current_factors if var not in f.vars]

            if not relevant:
                continue

            input_scopes = [f.vars for f in relevant]

            # Multiply all relevant factors
            prod_factor = relevant[0]
            for f in relevant[1:]:
                prod_factor = prod_factor.multiply(f)

            prod_scope = list(prod_factor.vars)

            # Marginalize out var
            new_factor = prod_factor.marginalize(var)
            current_factors = non_relevant + [new_factor]

            trace_step = EliminationTraceStep(
                step=step_idx,
                eliminated_var=var,
                input_factors_scopes=input_scopes,
                product_scope=prod_scope,
                result_scope=new_factor.vars,
                result_size=int(new_factor.values.size),
                description=f"Multiplied {len(relevant)} factor(s) over {prod_scope}, summed out '{var}'."
            )
            trace.append(trace_step.to_dict())

        # 4. Multiply all remaining factors together
        if not current_factors:
            raise RuntimeError("No factors remaining after elimination!")

        final_product = current_factors[0]
        for f in current_factors[1:]:
            final_product = final_product.multiply(f)

        # Marginalize out any non-target vars (if any remain)
        for v in list(final_product.vars):
            if v not in targets:
                final_product = final_product.marginalize(v)

        # Normalize to obtain posterior
        posterior = final_product.normalize()
        return posterior, trace
