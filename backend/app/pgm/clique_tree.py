"""
Probabilistic Graphical Models: Clique Tree (Junction Tree) & Belief Propagation.
Implements:
  - Moralization of DAG
  - Chordal graph triangulation via min-fill elimination
  - Maximal clique extraction
  - Maximum Spanning Tree via Kruskal's for Junction Tree (Running Intersection Property)
  - Factor assignment to clique families
  - Two-pass Sum-Product Belief Propagation (Collect & Distribute)
  - All-marginals query from a single calibration

Syllabus Topic: Clique Trees, Junction Trees, Message Passing, Belief Propagation.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
from backend.app.pgm.factor import Factor
from backend.app.pgm.network import BayesNet


class CliqueTreeNode:
    def __init__(self, clique_id: int, variables: Set[str]):
        self.clique_id = clique_id
        self.variables = frozenset(variables)
        self.neighbors: List[int] = []
        self.initial_potential: Optional[Factor] = None
        self.calibrated_potential: Optional[Factor] = None
        self.incoming_messages: Dict[int, Factor] = {}  # neighbor_id -> message Factor


class CliqueTree:
    """
    Junction Tree data structure satisfying the Running Intersection Property.
    """
    def __init__(self, net: BayesNet):
        self.net = net
        self.cliques: Dict[int, CliqueTreeNode] = {}
        self.edges: List[Tuple[int, int, Set[str]]] = []  # (u, v, sepset)
        self.sepsets: Dict[Tuple[int, int], Set[str]] = {}
        self.var_to_clique: Dict[str, int] = {}
        self._build_junction_tree()

    def _moralize(self) -> Dict[str, Set[str]]:
        """
        Moralizes DAG: connects co-parents and drops edge directions.
        """
        adj: Dict[str, Set[str]] = {n: set() for n in self.net.nodes}
        for n, pars in self.net.parents.items():
            for p in pars:
                adj[n].add(p)
                adj[p].add(n)
            # Connect all pairs of parents (marriage)
            for i in range(len(pars)):
                for j in range(i + 1, len(pars)):
                    p1, p2 = pars[i], pars[j]
                    adj[p1].add(p2)
                    adj[p2].add(p1)
        return adj

    def _triangulate(self, moral_graph: Dict[str, Set[str]]) -> Tuple[Dict[str, Set[str]], List[Set[str]]]:
        """
        Triangulates moral graph via min-fill heuristic and extracts maximal cliques.
        """
        graph = {n: set(moral_graph[n]) for n in moral_graph}
        remaining = set(self.net.nodes)
        cliques_found: List[Set[str]] = []

        # Elimination order
        while remaining:
            best_node = None
            min_fill = float("inf")
            for node in remaining:
                nbrs = graph[node] & remaining
                fill = 0
                nbr_list = list(nbrs)
                for i in range(len(nbr_list)):
                    for j in range(i + 1, len(nbr_list)):
                        if nbr_list[j] not in graph[nbr_list[i]]:
                            fill += 1
                if fill < min_fill:
                    min_fill = fill
                    best_node = node

            assert best_node is not None
            # The node and its remaining neighbors form a clique
            clique = (graph[best_node] & remaining) | {best_node}
            cliques_found.append(set(clique))

            # Add fill edges to graph
            nbrs = list(graph[best_node] & remaining)
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    u, v = nbrs[i], nbrs[j]
                    graph[u].add(v)
                    graph[v].add(u)

            remaining.remove(best_node)

        # Retain only maximal cliques (remove subsets)
        maximal_cliques: List[Set[str]] = []
        for c in cliques_found:
            # Check if strictly contained in any other clique
            is_subset = False
            for other in cliques_found:
                if c != other and c.issubset(other):
                    is_subset = True
                    break
            if not is_subset and c not in maximal_cliques:
                maximal_cliques.append(c)

        return graph, maximal_cliques

    def _build_junction_tree(self):
        """
        Constructs Junction Tree from maximal cliques using Maximum Spanning Tree.
        """
        moral = self._moralize()
        _, max_cliques = self._triangulate(moral)

        for i, c_vars in enumerate(max_cliques):
            self.cliques[i] = CliqueTreeNode(i, c_vars)

        # Form candidate edges between all pairs of cliques with weight = |C_i \cap C_j|
        candidate_edges = []
        n_cliques = len(max_cliques)
        for i in range(n_cliques):
            for j in range(i + 1, n_cliques):
                sep = max_cliques[i] & max_cliques[j]
                weight = len(sep)
                if weight > 0:
                    candidate_edges.append((weight, i, j, sep))

        # Sort descending by weight (Max Spanning Tree for RIP)
        candidate_edges.sort(key=lambda x: x[0], reverse=True)

        # Kruskal's algorithm
        parent_ds = list(range(n_cliques))

        def find(x: int) -> int:
            if parent_ds[x] != x:
                parent_ds[x] = find(parent_ds[x])
            return parent_ds[x]

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent_ds[rx] = ry
                return True
            return False

        edges_in_tree = []
        for weight, u, v, sep in candidate_edges:
            if union(u, v):
                edges_in_tree.append((u, v, sep))
                self.cliques[u].neighbors.append(v)
                self.cliques[v].neighbors.append(u)
                self.sepsets[(u, v)] = sep
                self.sepsets[(v, u)] = sep

        # If disconnected (independent components), connect with empty sepsets
        for i in range(n_cliques):
            for j in range(i + 1, n_cliques):
                if union(i, j):
                    edges_in_tree.append((i, j, set()))
                    self.cliques[i].neighbors.append(j)
                    self.cliques[j].neighbors.append(i)
                    self.sepsets[(i, j)] = set()
                    self.sepsets[(j, i)] = set()

        self.edges = edges_in_tree

        # Map each variable to a clique containing it
        for var in self.net.nodes:
            for c_id, node in self.cliques.items():
                if var in node.variables:
                    self.var_to_clique[var] = c_id
                    break

    def calibrate(self, evidence: Optional[Dict[str, str]] = None) -> None:
        """
        Initializes potentials and performs two-pass belief propagation.
        """
        if evidence is None:
            evidence = {}

        evidence_indices: Dict[str, int] = {
            v: self.net.states[v].index(val) for v, val in evidence.items()
        }

        # 1. Initialize potentials with unity
        for c_id, c_node in self.cliques.items():
            vars_list = sorted(list(c_node.variables))
            cards_list = [self.net.cards[v] for v in vars_list]
            ones = np.ones(cards_list, dtype=np.float64)
            c_node.initial_potential = Factor(vars_list, cards_list, ones)
            c_node.incoming_messages = {}

        # 2. Assign each CPT to exactly one clique that covers Family(X) = {X} U Parents(X)
        for node_name in self.net.nodes:
            fam = set(self.net.parents[node_name]) | {node_name}
            assigned = False
            for c_id, c_node in self.cliques.items():
                if fam.issubset(c_node.variables):
                    c_node.initial_potential = c_node.initial_potential.multiply(self.net.cpts[node_name])
                    assigned = True
                    break
            if not assigned:
                raise RuntimeError(f"No clique contains family of {node_name}")

        # 3. Reduce initial potentials by evidence
        for c_id, c_node in self.cliques.items():
            c_node.initial_potential = c_node.initial_potential.reduce(evidence_indices)

        # 4. Message passing (Sum-Product Belief Propagation)
        # Choose root node (e.g., clique 0)
        root = 0
        visited = set()

        def send_message(from_id: int, to_id: int):
            # Compute message: mu_{i->j}(S) = sum_{C_i \ S} (psi_i * prod_{k in N(i) \ {j}} mu_{k->i})
            pot = self.cliques[from_id].initial_potential.copy()
            for k in self.cliques[from_id].neighbors:
                if k != to_id and k in self.cliques[from_id].incoming_messages:
                    pot = pot.multiply(self.cliques[from_id].incoming_messages[k])

            sepset = self.sepsets.get((from_id, to_id), set())
            # Sum out all variables in pot that are not in sepset
            for v in list(pot.vars):
                if v not in sepset:
                    pot = pot.marginalize(v)

            self.cliques[to_id].incoming_messages[from_id] = pot

        # Pass 1: Collect towards root (post-order traversal)
        def collect(curr: int, parent: Optional[int]):
            visited.add(curr)
            for nbr in self.cliques[curr].neighbors:
                if nbr != parent:
                    collect(nbr, curr)
                    send_message(nbr, curr)

        collect(root, None)

        # Pass 2: Distribute outwards from root (pre-order traversal)
        def distribute(curr: int, parent: Optional[int]):
            for nbr in self.cliques[curr].neighbors:
                if nbr != parent:
                    send_message(curr, nbr)
                    distribute(nbr, curr)

        distribute(root, None)

        # 5. Compute calibrated belief for each clique
        for c_id, c_node in self.cliques.items():
            bel = c_node.initial_potential.copy()
            for nbr in c_node.neighbors:
                bel = bel.multiply(c_node.incoming_messages[nbr])
            c_node.calibrated_potential = bel

    def query(self, target_var: str, evidence: Optional[Dict[str, str]] = None) -> Factor:
        """
        Returns posterior Factor for target_var using the calibrated clique tree.
        """
        self.calibrate(evidence)
        # Find clique containing target_var
        for c_id, c_node in self.cliques.items():
            if target_var in c_node.calibrated_potential.vars:
                res = c_node.calibrated_potential.copy()
                for v in list(res.vars):
                    if v != target_var:
                        res = res.marginalize(v)
                return res.normalize()

        # If target was in evidence, return deterministic indicator
        if evidence and target_var in evidence:
            val = evidence[target_var]
            idx = self.net.states[target_var].index(val)
            card = self.net.cards[target_var]
            arr = np.zeros(card, dtype=np.float64)
            arr[idx] = 1.0
            return Factor([target_var], [card], arr)

        raise KeyError(f"Variable '{target_var}' not found in any calibrated potential")

    def get_tree_structure(self) -> Dict[str, Any]:
        """
        Returns clique tree metadata for UI rendering.
        """
        nodes = []
        for c_id, c_node in self.cliques.items():
            nodes.append({
                "id": str(c_id),
                "label": f"Clique {c_id}: {{{', '.join(sorted(c_node.variables))}}}",
                "variables": sorted(list(c_node.variables))
            })
        edges = []
        for u, v, sep in self.edges:
            edges.append({
                "source": str(u),
                "target": str(v),
                "sepset": sorted(list(sep)),
                "label": f"{{{', '.join(sorted(sep))}}}"
            })
        return {"nodes": nodes, "edges": edges}
