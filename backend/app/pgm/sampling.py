"""
Probabilistic Graphical Models: Approximate Inference by Stochastic Sampling.
Implements:
  1. Rejection Sampling (forward sampling with evidence rejection)
  2. Likelihood Weighting (forward sampling weighting by evidence likelihood)
  3. Gibbs Sampling (MCMC with Markov Blanket conditional distributions)
  4. Effective Sample Size (ESS) estimation
  5. Convergence analysis: KL-divergence vs Sample Size (N)

Syllabus Topic: Approximate Inference, Monte Carlo Methods, MCMC, Gibbs Sampling.
"""
from __future__ import annotations
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from backend.app.pgm.network import BayesNet
from backend.app.pgm.variable_elimination import VariableElimination


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes D_KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
    """
    p_safe = np.clip(p, eps, 1.0)
    p_safe /= p_safe.sum()
    q_safe = np.clip(q, eps, 1.0)
    q_safe /= q_safe.sum()
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


class Sampler:
    def __init__(self, net: BayesNet):
        self.net = net
        self.ve = VariableElimination(net)

    def rejection_sampling(
        self,
        targets: List[str],
        evidence: Dict[str, str],
        n_samples: int = 10000,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, Any]:
        """
        Samples full assignments from prior in topological order;
        rejects assignments inconsistent with evidence.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        accepted_counts: Dict[str, np.ndarray] = {
            t: np.zeros(self.net.cards[t], dtype=np.float64) for t in targets
        }
        accepted_total = 0

        for _ in range(n_samples):
            assignment: Dict[str, int] = {}
            consistent = True

            for n in self.net.topological_order:
                pars = self.net.parents[n]
                if len(pars) == 0:
                    probs = self.net.nodes_spec[n]["cpt"][""]
                else:
                    key = "|".join(self.net.states[p][assignment[p]] for p in pars)
                    probs = self.net.nodes_spec[n]["cpt"][key]

                idx = rng.choice(self.net.cards[n], p=probs)
                assignment[n] = idx

                if n in evidence:
                    ev_idx = self.net.states[n].index(evidence[n])
                    if idx != ev_idx:
                        consistent = False
                        break

            if consistent:
                accepted_total += 1
                for t in targets:
                    accepted_counts[t][assignment[t]] += 1.0

        acceptance_rate = accepted_total / n_samples
        posteriors = {}
        for t in targets:
            if accepted_total > 0:
                dist = accepted_counts[t] / accepted_total
            else:
                dist = np.ones(self.net.cards[t]) / self.net.cards[t]
            posteriors[t] = {self.net.states[t][i]: float(dist[i]) for i in range(self.net.cards[t])}

        return {
            "method": "rejection",
            "samples_requested": n_samples,
            "samples_accepted": accepted_total,
            "acceptance_rate": round(float(acceptance_rate), 6),
            "effective_sample_size": accepted_total,
            "posteriors": posteriors
        }

    def likelihood_weighting(
        self,
        targets: List[str],
        evidence: Dict[str, str],
        n_samples: int = 10000,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, Any]:
        """
        Samples unobserved variables from P(X | parents(X));
        sets observed variables to evidence and accumulates weight = prod P(e | parents(e)).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        weights = np.zeros(n_samples, dtype=np.float64)
        target_indices = {t: np.zeros(n_samples, dtype=np.int32) for t in targets}

        for s_idx in range(n_samples):
            assignment: Dict[str, int] = {}
            weight = 1.0

            for n in self.net.topological_order:
                pars = self.net.parents[n]
                if len(pars) == 0:
                    probs = self.net.nodes_spec[n]["cpt"][""]
                else:
                    key = "|".join(self.net.states[p][assignment[p]] for p in pars)
                    probs = self.net.nodes_spec[n]["cpt"][key]

                if n in evidence:
                    ev_idx = self.net.states[n].index(evidence[n])
                    assignment[n] = ev_idx
                    weight *= probs[ev_idx]
                else:
                    idx = rng.choice(self.net.cards[n], p=probs)
                    assignment[n] = idx

            weights[s_idx] = weight
            for t in targets:
                target_indices[t][s_idx] = assignment[t]

        sum_w = float(np.sum(weights))
        posteriors = {}
        for t in targets:
            card = self.net.cards[t]
            counts = np.zeros(card, dtype=np.float64)
            for i in range(card):
                counts[i] = np.sum(weights[target_indices[t] == i])
            dist = counts / sum_w if sum_w > 0 else np.ones(card) / card
            posteriors[t] = {self.net.states[t][i]: float(dist[i]) for i in range(card)}

        # Kish's Effective Sample Size: ESS = (sum w_i)^2 / sum (w_i^2)
        sum_sq = float(np.sum(weights**2))
        ess = (sum_w**2) / sum_sq if sum_sq > 0 else 0.0

        return {
            "method": "likelihood_weighting",
            "samples_requested": n_samples,
            "total_weight": float(sum_w),
            "effective_sample_size": round(float(ess), 2),
            "posteriors": posteriors
        }

    def gibbs_sampling(
        self,
        targets: List[str],
        evidence: Dict[str, str],
        n_samples: int = 10000,
        burn_in: int = 1000,
        thinning: int = 2,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, Any]:
        """
        Gibbs MCMC sampling using Markov Blanket full conditionals:
        P(X_i | MB(X_i)) proportional to P(X_i | Pa(X_i)) * prod_{Y in Ch(X_i)} P(Y | Pa(Y)).
        """
        if rng is None:
            rng = np.random.default_rng(42)

        unobserved = [n for n in self.net.nodes if n not in evidence]

        # Initialize state uniformly or from prior
        current_state: Dict[str, int] = {}
        for n in self.net.nodes:
            if n in evidence:
                current_state[n] = self.net.states[n].index(evidence[n])
            else:
                current_state[n] = rng.integers(0, self.net.cards[n])

        # Precompute children and Markov Blanket components
        counts = {t: np.zeros(self.net.cards[t], dtype=np.float64) for t in targets}
        recorded_samples = 0

        total_steps = burn_in + n_samples * thinning

        for step in range(total_steps):
            # Update each unobserved variable in sequence
            for var in unobserved:
                card = self.net.cards[var]
                probs = np.zeros(card, dtype=np.float64)

                pars = self.net.parents[var]
                children = self.net.children[var]

                for candidate_val in range(card):
                    current_state[var] = candidate_val
                    # Term 1: P(var | Pa(var))
                    if len(pars) == 0:
                        p_var = self.net.nodes_spec[var]["cpt"][""][candidate_val]
                    else:
                        p_key = "|".join(self.net.states[p][current_state[p]] for p in pars)
                        p_var = self.net.nodes_spec[var]["cpt"][p_key][candidate_val]

                    # Term 2: prod_{Y in Ch(var)} P(Y | Pa(Y))
                    p_children = 1.0
                    for ch in children:
                        ch_pars = self.net.parents[ch]
                        ch_key = "|".join(self.net.states[p][current_state[p]] for p in ch_pars)
                        p_children *= self.net.nodes_spec[ch]["cpt"][ch_key][current_state[ch]]

                    probs[candidate_val] = p_var * p_children

                p_sum = probs.sum()
                if p_sum > 0:
                    probs /= p_sum
                else:
                    probs = np.ones(card) / card

                chosen_val = rng.choice(card, p=probs)
                current_state[var] = chosen_val

            # Record after burn-in with thinning
            if step >= burn_in and ((step - burn_in) % thinning == 0):
                recorded_samples += 1
                for t in targets:
                    counts[t][current_state[t]] += 1.0

        posteriors = {}
        for t in targets:
            dist = counts[t] / recorded_samples if recorded_samples > 0 else np.ones(self.net.cards[t]) / self.net.cards[t]
            posteriors[t] = {self.net.states[t][i]: float(dist[i]) for i in range(self.net.cards[t])}

        return {
            "method": "gibbs",
            "samples_requested": n_samples,
            "burn_in": burn_in,
            "thinning": thinning,
            "effective_sample_size": recorded_samples,
            "posteriors": posteriors
        }

    def run_convergence_experiment(
        self,
        evidence: Dict[str, str],
        target: str = "Severity",
        sample_sizes: Optional[List[int]] = None,
        repeats: int = 20,
        rng_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs convergence experiment across sample sizes N in {100, 300, 1k, 3k, 10k, 30k, 100k}.
        Computes mean and std of KL(exact || estimate) and max absolute error for all 3 methods.
        """
        if sample_sizes is None:
            sample_sizes = [100, 300, 1000, 3000, 10000, 30000, 100000]

        # Exact ground truth posterior via Variable Elimination
        exact_factor, _ = self.ve.query([target], evidence)
        exact_dist = exact_factor.get_distribution({target: self.net.states[target]})
        exact_vec = np.array([exact_dist[s] for s in self.net.states[target]], dtype=np.float64)

        rng = np.random.default_rng(rng_seed)
        results = {
            "sample_sizes": sample_sizes,
            "exact_distribution": {s: round(float(exact_dist[s]), 4) for s in self.net.states[target]},
            "rejection": {"kl_mean": [], "kl_std": [], "max_error_mean": []},
            "lw": {"kl_mean": [], "kl_std": [], "max_error_mean": []},
            "gibbs": {"kl_mean": [], "kl_std": [], "max_error_mean": []}
        }

        for n in sample_sizes:
            # Rejection
            rej_kls, rej_errs = [], []
            lw_kls, lw_errs = [], []
            gibbs_kls, gibbs_errs = [], []

            for _ in range(repeats):
                # Rejection
                rej_res = self.rejection_sampling([target], evidence, n_samples=n, rng=rng)
                r_vec = np.array([rej_res["posteriors"][target][s] for s in self.net.states[target]])
                rej_kls.append(kl_divergence(exact_vec, r_vec))
                rej_errs.append(float(np.max(np.abs(exact_vec - r_vec))))

                # LW
                lw_res = self.likelihood_weighting([target], evidence, n_samples=n, rng=rng)
                l_vec = np.array([lw_res["posteriors"][target][s] for s in self.net.states[target]])
                lw_kls.append(kl_divergence(exact_vec, l_vec))
                lw_errs.append(float(np.max(np.abs(exact_vec - l_vec))))

                # Gibbs (use small burn-in for small n, larger for big n)
                burn = min(500, max(50, n // 10))
                gibbs_res = self.gibbs_sampling([target], evidence, n_samples=n, burn_in=burn, thinning=1, rng=rng)
                g_vec = np.array([gibbs_res["posteriors"][target][s] for s in self.net.states[target]])
                gibbs_kls.append(kl_divergence(exact_vec, g_vec))
                gibbs_errs.append(float(np.max(np.abs(exact_vec - g_vec))))

            results["rejection"]["kl_mean"].append(round(float(np.mean(rej_kls)), 6))
            results["rejection"]["kl_std"].append(round(float(np.std(rej_kls)), 6))
            results["rejection"]["max_error_mean"].append(round(float(np.mean(rej_errs)), 6))

            results["lw"]["kl_mean"].append(round(float(np.mean(lw_kls)), 6))
            results["lw"]["kl_std"].append(round(float(np.std(lw_kls)), 6))
            results["lw"]["max_error_mean"].append(round(float(np.mean(lw_errs)), 6))

            results["gibbs"]["kl_mean"].append(round(float(np.mean(gibbs_kls)), 6))
            results["gibbs"]["kl_std"].append(round(float(np.std(gibbs_kls)), 6))
            results["gibbs"]["max_error_mean"].append(round(float(np.mean(gibbs_errs)), 6))

        return results
