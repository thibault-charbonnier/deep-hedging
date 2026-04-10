"""
Deep Hedging of Derivatives Using Reinforcement Learning
=========================================================
Reproduction and extension of Cao, Chen, Hull & Poulos (2021).
ML for Finance — M2MO / ENSAE / Mastère FGR, 2025-2026.

Structure
─────────
  PART A — Faithful replication of the paper
    A.1  GBM,  T=1 month,  varying rebalancing frequency    (Exhibit 3)
    A.2  GBM,  T=3 months, varying rebalancing frequency    (Exhibit 4)
    A.3  SABR, T=1 month,  3 benchmarks, varying frequency  (Exhibit 6)
    A.4  SABR, T=3 months, 3 benchmarks, varying frequency  (Exhibit 7)
    A.5  Over/under-hedging analysis                         (Exhibit 5)

  PART B — Extensions beyond the paper
    B.1  SVJ stochastic volatility with jumps
    B.2  Longer maturities (6 months, 1 year)
    B.3  Agent comparison: DQN vs DoubleDQN vs DeepDPG
    B.4  Cash Flow vs Accounting P&L  (Sections 3.1-3.3)
"""
import logging, copy
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as sp_stats
from rich.logging import RichHandler

from src.orchestrator import Orchestrator
from src.valuation import BSValuation
from src.benchmark import SABRPractitionerDeltaBenchmark, BartlettDeltaBenchmark
from src.hedging_result import HedgingResult, EpisodeResult
from src.utils.enums import AgentType, BenchmarkType, ProcessType
from src.utils.helpers import json_to_dict

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
def reset_seed():
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
reset_seed()

# ── Logging & output ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)], force=True)
logger = logging.getLogger(__name__)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)


# ======================================================================
#  HELPERS
# ======================================================================

def total_costs(res, split):
    return [sum(e.costs) for e in res.episodes[split]]


def metrics(costs, lam=1.5):
    """Mean, std, Y(0), 95 % CI on mean, bootstrap CI on Y(0).
    Bug-fix V7: bootstrap uses the SAME sample for mean and std."""
    a = np.asarray(costs); n = len(a)
    m, s = float(a.mean()), float(a.std(ddof=1))
    t_c = float(sp_stats.t.ppf(0.975, df=n - 1))
    rng = np.random.RandomState(0)
    boot_y = np.empty(5000)
    for b in range(5000):
        sample = a[rng.randint(0, n, n)]
        boot_y[b] = sample.mean() + lam * sample.std(ddof=1)
    return {"mean": m, "std": s, "Y": m + lam * s,
            "mean_ci": t_c * s / np.sqrt(n),
            "Y_lo": float(np.percentile(boot_y, 2.5)),
            "Y_hi": float(np.percentile(boot_y, 97.5))}


def bs_price(cfg):
    bs = BSValuation(cfg["derivative"]["strike"], cfg["simulation"]["maturity"],
        cfg["derivative"].get("rf_rate", 0), cfg["derivative"].get("div_rate", 0),
        cfg["derivative"].get("option_type", "call"))
    p, _ = bs.price_and_delta(cfg["simulation"]["S0"], 0,
                               cfg["simulation"]["gbm"]["sigma"])
    return abs(p)


def convergence(train_res, w=500, thr=0.05):
    c = [sum(e.costs) for e in train_res.episodes["train"]]; n = len(c)
    if n < 2 * w: w = max(n // 4, 10)
    prev, last = float(np.mean(c[-2*w:-w])), float(np.mean(c[-w:]))
    rc = abs(last - prev) / (abs(prev) + 1e-12)
    return {"ok": rc < thr, "rc": rc, "prev": prev, "last": last, "w": w}


def welch(a, b):
    t, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p)}


def sig_stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def freq_map(T):
    """Paper rebalancing frequencies for a given maturity."""
    return {"weekly": max(int(round(52*T)), 3),
            "3-day":  max(int(round(84*T)), 5),
            "2-day":  max(int(round(126*T)), 5),
            "daily":  max(int(round(252*T)), 5)}


# ── Runners ──────────────────────────────────────────────────────────

def run(cfg, proc, agent, bench):
    """Train + eval with Accounting P&L env, single benchmark."""
    reset_seed()
    o = Orchestrator(cfg, proc, agent, bench)
    tr = o.train(); ea = o.test(); eb = o.test_benchmark()
    lam = float(cfg["hedging_agent"].get("risk_lambda", 1.5))
    ac, bc = total_costs(ea, "eval_agent"), total_costs(eb, "eval_benchmark")
    am, bm = metrics(ac, lam), metrics(bc, lam)
    imp = 100 * (bm["Y"] - am["Y"]) / bm["Y"] if bm["Y"] != 0 else 0
    cv = convergence(tr); tt = welch(ac, bc)
    logger.info(f"  Conv: {'✅' if cv['ok'] else '⚠️'} Δ={cv['rc']:.3%} | "
                f"Welch p={tt['p']:.2e}({sig_stars(tt['p'])})")
    return {"agent": am, "bench": bm, "imp": imp, "tr": tr, "ea": ea, "eb": eb,
            "ac": ac, "bc": bc, "cv": cv, "tt": tt, "runner": o}


def run_multi_bench(cfg, proc, agent):
    """Train once, then eval against Practitioner Δ and Bartlett Δ."""
    reset_seed()
    o = Orchestrator(cfg, proc, agent, BenchmarkType.BsDelta)
    tr = o.train(); ea = o.test()
    lam = float(cfg["hedging_agent"].get("risk_lambda", 1.5))
    ac = total_costs(ea, "eval_agent"); am = metrics(ac, lam)
    cv = convergence(tr)
    logger.info(f"  Conv: {'✅' if cv['ok'] else '⚠️'} Δ={cv['rc']:.3%}")
    benchmarks = {}
    for bname, btype in [("Practitioner Δ", BenchmarkType.SABRPractitionerDelta),
                         ("Bartlett Δ",     BenchmarkType.BartlettDelta)]:
        eb = o.test_benchmark(benchmark_override=btype.value(cfg))
        bc = total_costs(eb, "eval_benchmark"); bm = metrics(bc, lam)
        imp = 100 * (bm["Y"] - am["Y"]) / bm["Y"] if bm["Y"] != 0 else 0
        benchmarks[bname] = {"metrics": bm, "imp": imp, "tt": welch(ac, bc), "costs": bc}
    return {"agent": am, "benchmarks": benchmarks, "tr": tr, "ea": ea,
            "ac": ac, "cv": cv}


def ptable(rows, title):
    df = pd.DataFrame(rows)
    logger.info(f"\n{'─'*85}\n  {title}\n{'─'*85}")
    logger.info(df.to_string(index=False, float_format="%.1f"))
    logger.info(f"{'─'*85}\n")
    return df


# ======================================================================
#  CASH FLOW ENV  (Paper Section 3.2)
# ======================================================================

class CashFlowHedgingEnv:
    """
    Cash Flow formulation — Section 3.2.

    R_{i+1} = S_{i+1}(H_i - H_{i+1}) - κ|S_{i+1}(H_{i+1} - H_i)|
    Initial:  -S_0 H_0 - κ|S_0 H_0|
    Final:    S_n H_n - κ|S_n H_n| + payoff

    "The accounting P&L approach gives much better results than the
     cash flow approach" — Section 3.3.
    """
    def __init__(self, config):
        self.kappa = float(config["hedging_env"]["transaction_cost"])
        self.position_sign = float(config["hedging_env"]["position_sign"])
        self.option_type = config.get("derivative", {}).get("option_type", "call")
        self.K = float(config["derivative"]["strike"])
        self.maturity = float(config["simulation"]["maturity"])
        self.sigma_ref = float(config["simulation"]["gbm"]["sigma"])

    def setup_env(self, path_data):
        if isinstance(path_data, dict):
            self.path_dict = {k: np.asarray(v, dtype=float) for k, v in path_data.items()}
            self.path_data = self.path_dict["S"]
        else:
            self.path_data = np.asarray(path_data, dtype=float)
            self.path_dict = {"S": self.path_data}
        if "sigma" in self.path_dict:
            self._vol = self.path_dict["sigma"]
        elif "variance" in self.path_dict:
            self._vol = np.sqrt(np.maximum(self.path_dict["variance"], 1e-10))
        else:
            self._vol = np.full_like(self.path_data, self.sigma_ref)
        self.n_steps = len(self.path_data) - 1
        self.times = np.linspace(0.0, self.maturity, len(self.path_data))
        self.i = 0; self.h_prev = 0.0; self.is_first = True
        self.ep_reward = 0.0; self.ep_cost = 0.0
        return self._state(0, 0.0)

    def step(self, hedge):
        hedge = float(hedge)
        i = self.i
        s_t, s_next = float(self.path_data[i]), float(self.path_data[i + 1])
        done = i == self.n_steps - 1

        if self.is_first:
            tc = self.kappa * s_t * abs(hedge)
            reward = -(s_t * hedge + tc)
            self.is_first = False
        else:
            tc = self.kappa * s_next * abs(hedge - self.h_prev)
            reward = s_next * (self.h_prev - hedge) - tc

        liq = 0.0
        if done:
            liq = self.kappa * s_next * abs(hedge)
            liq_val = s_next * hedge - liq
            # Option payoff for the hedger:
            #   Short call (position_sign=-1): hedger PAYS max(S-K, 0)
            #     → payoff = -1 * max(S-K, 0) = negative reward (cost)
            #   Short put  (position_sign=-1): hedger PAYS max(K-S, 0)
            if self.option_type == "call":
                payoff = self.position_sign * max(s_next - self.K, 0.0)
            else:
                payoff = self.position_sign * max(self.K - s_next, 0.0)
            reward += liq_val + payoff

        self.ep_reward += reward; self.ep_cost += -reward
        self.i += 1; self.h_prev = 0.0 if done else hedge
        info = {"reward": reward, "cost": -reward, "trade_cost": tc,
                "liquidation_cost": liq, "episode_reward": self.ep_reward,
                "episode_cost": self.ep_cost, "spot_t": s_t, "spot_next": s_next,
                "hedge": hedge}
        return self._state(self.i, self.h_prev), reward, done, info

    def _state(self, step, h):
        idx = min(step, len(self.path_data) - 1)
        t = self.times[min(step, len(self.times) - 1)]
        s, v = self.path_data[idx], self._vol[idx]
        ttm = max(self.maturity - t, 0.0)
        return np.asarray([h, np.log(s / self.K),
                           ttm / self.maturity if self.maturity > 0 else 0.0,
                           v / self.sigma_ref], dtype=float)


def run_cashflow(cfg, proc_type, agent_type):
    """Train + eval with Cash Flow env."""
    reset_seed()
    process = proc_type.value(cfg["simulation"])
    agent = agent_type.value(cfg["hedging_agent"])
    env = CashFlowHedgingEnv(cfg)
    n_tr = int(cfg["training_schedule"]["train_episodes"])
    n_ev = int(cfg["training_schedule"]["eval_episodes"])
    tr_paths = process.simulate_paths(n_tr)
    ev_paths = process.simulate_paths(n_ev)
    ep = lambda paths, i: {k: v[i] for k, v in paths.items()}

    agent.set_train_mode()
    tr = HedgingResult()
    for e in range(n_tr):
        s = env.setup_env(ep(tr_paths, e)); done = False
        er = EpisodeResult(split="train", episode_idx=e,
                           times=env.times, path_data=ep(tr_paths, e))
        while not done:
            a = agent.act(s, eval_mode=False)
            ns, r, done, info = env.step(a)
            agent.store_transition(s, a, r, ns, done)
            loss = agent.learn()
            er.add_step(action=a, info=info, loss=loss); s = ns
        tr.add_episode(er, type="train")

    agent.set_eval_mode()
    ev = HedgingResult()
    for e in range(n_ev):
        s = env.setup_env(ep(ev_paths, e)); done = False
        er = EpisodeResult(split="eval_agent", episode_idx=e,
                           times=env.times, path_data=ep(ev_paths, e))
        while not done:
            a = agent.act(s, eval_mode=True)
            s, _, done, info = env.step(a)
            er.add_step(action=a, info=info)
        ev.add_episode(er, type="eval_agent")

    lam = float(cfg["hedging_agent"].get("risk_lambda", 1.5))
    costs = total_costs(ev, "eval_agent")
    return {"metrics": metrics(costs, lam), "costs": costs,
            "train_res": tr, "cv": convergence(tr)}


# ======================================================================
#  PART A — PAPER REPLICATION
# ======================================================================

def part_A1_A2(base_cfg, T, exhibit_name):
    """GBM, varying rebalancing freq — Exhibits 3 & 4."""
    logger.info("=" * 65)
    logger.info(f"PART A — {exhibit_name}: GBM, T={T*12:.0f}m, "
                f"Short ATM Call, S₀=100, σ=20%, μ=5%, κ=1%")
    logger.info("=" * 65)
    V0 = None; rows = []
    for label, ns in freq_map(T).items():
        logger.info(f"  ► {label} (n={ns})")
        cfg = copy.deepcopy(base_cfg)
        cfg["simulation"]["maturity"] = T; cfg["simulation"]["n_steps"] = ns
        if V0 is None: V0 = bs_price(cfg)
        r = run(cfg, ProcessType.GBM, AgentType.DeepDPG, BenchmarkType.BsDelta)
        rows.append({"Rebal": label,
            "Δ Mean%": r["bench"]["mean"]/V0*100,
            "Δ Std%":  r["bench"]["std"]/V0*100,
            "RL Mean%": r["agent"]["mean"]/V0*100,
            "RL Std%":  r["agent"]["std"]/V0*100,
            "Y improv%": r["imp"],
            "p-value": f"{r['tt']['p']:.2e}",
            "conv": "✅" if r["cv"]["ok"] else "⚠️"})
    return ptable(rows, f"{exhibit_name} — GBM, T={T*12:.0f}m, "
                        f"κ=1%, μ=5%, σ=20%, r=0, q=0")


def part_A3_A4(base_cfg, T, exhibit_name):
    """SABR, 3 benchmarks, varying freq — Exhibits 6 & 7."""
    logger.info("=" * 65)
    logger.info(f"PART A — {exhibit_name}: SABR β=1, T={T*12:.0f}m, "
                f"ρ=-0.4, ν=0.6, σ₀=20%")
    logger.info("=" * 65)
    V0 = None; rows = []
    for label, ns in freq_map(T).items():
        logger.info(f"  ► {label} (n={ns})")
        cfg = copy.deepcopy(base_cfg)
        cfg["simulation"]["maturity"] = T; cfg["simulation"]["n_steps"] = ns
        if V0 is None: V0 = bs_price(cfg)
        res = run_multi_bench(cfg, ProcessType.SABR, AgentType.DeepDPG)
        row = {"Rebal": label,
               "RL Mean%": res["agent"]["mean"]/V0*100,
               "RL Std%":  res["agent"]["std"]/V0*100}
        for bname, bd in res["benchmarks"].items():
            bm = bd["metrics"]
            row[f"{bname} Mean%"] = bm["mean"]/V0*100
            row[f"{bname} Std%"]  = bm["std"]/V0*100
            row[f"Y vs {bname}%"] = bd["imp"]
        rows.append(row)
    return ptable(rows, f"{exhibit_name} — SABR(β=1,ρ=-0.4,ν=0.6), "
                        f"T={T*12:.0f}m, κ=1%")


# ======================================================================
#  PART B — EXTENSIONS
# ======================================================================

def part_B1_svj(base_cfg):
    """SVJ process."""
    logger.info("=" * 65)
    logger.info("PART B.1 — SVJ (Heston + Poisson jumps) vs GBM")
    logger.info("=" * 65)
    rows = []
    for Tl, T in [("1m", 1/12), ("3m", 3/12)]:
        cfg = copy.deepcopy(base_cfg)
        cfg["simulation"]["maturity"] = T
        cfg["simulation"]["n_steps"] = max(int(round(252*T)), 5)
        V0 = bs_price(cfg)
        for pn, pt in [("GBM", ProcessType.GBM), ("SVJ", ProcessType.SVJ)]:
            logger.info(f"  ► {pn}, T={Tl}")
            r = run(cfg, pt, AgentType.DeepDPG, BenchmarkType.BsDelta)
            rows.append({"T": Tl, "Process": pn,
                "Δ Mean%": r["bench"]["mean"]/V0*100,
                "Δ Std%":  r["bench"]["std"]/V0*100,
                "RL Mean%": r["agent"]["mean"]/V0*100,
                "RL Std%":  r["agent"]["std"]/V0*100,
                "Y improv%": r["imp"],
                "conv": "✅" if r["cv"]["ok"] else "⚠️"})
    return ptable(rows, "B.1 — SVJ vs GBM | DeepDPG, daily rebal, κ=1%")


def part_B2_maturities(base_cfg):
    """Longer maturities."""
    logger.info("=" * 65)
    logger.info("PART B.2 — Maturities: 1m / 3m / 6m / 1y")
    logger.info("=" * 65)
    rows = []
    for label, T in [("1m",1/12),("3m",3/12),("6m",6/12),("1y",1.0)]:
        ns = max(int(round(252*T)), 5)
        logger.info(f"  ► {label} (n={ns})")
        cfg = copy.deepcopy(base_cfg)
        cfg["simulation"]["maturity"] = T; cfg["simulation"]["n_steps"] = ns
        V0 = bs_price(cfg)
        r = run(cfg, ProcessType.GBM, AgentType.DeepDPG, BenchmarkType.BsDelta)
        rows.append({"Maturity": label,
            "Δ Mean%": r["bench"]["mean"]/V0*100,
            "Δ Std%":  r["bench"]["std"]/V0*100,
            "RL Mean%": r["agent"]["mean"]/V0*100,
            "RL Std%":  r["agent"]["std"]/V0*100,
            "Y improv%": r["imp"],
            "conv": "✅" if r["cv"]["ok"] else "⚠️"})
    return ptable(rows, "B.2 — Maturity comparison | GBM, DeepDPG, daily, κ=1%")


def part_B3_agents(base_cfg):
    """DQN vs DoubleDQN vs DeepDPG."""
    logger.info("=" * 65)
    logger.info("PART B.3 — DQN vs DoubleDQN vs DeepDPG")
    logger.info("=" * 65)
    cfg = copy.deepcopy(base_cfg)
    cfg["simulation"]["maturity"] = 3/12; cfg["simulation"]["n_steps"] = 63
    V0 = bs_price(cfg); rows = []
    for an, at in [("DQN",AgentType.DQN), ("DoubleDQN",AgentType.DoubleDQN),
                   ("DeepDPG",AgentType.DeepDPG)]:
        logger.info(f"  ► {an}")
        r = run(cfg, ProcessType.GBM, at, BenchmarkType.BsDelta)
        rows.append({"Agent": an,
            "Mean%":  r["agent"]["mean"]/V0*100,
            "±CI%":   r["agent"]["mean_ci"]/V0*100,
            "Std%":   r["agent"]["std"]/V0*100,
            "Y(0)%":  r["agent"]["Y"]/V0*100,
            "Y 95%CI": f"[{r['agent']['Y_lo']/V0*100:.1f}, {r['agent']['Y_hi']/V0*100:.1f}]",
            "Y vs Δ%": r["imp"],
            "conv": "✅" if r["cv"]["ok"] else "⚠️"})
    return ptable(rows, "B.3 — Agent comparison | GBM T=3m daily, κ=1%\n"
                        "  All agents use dual Q-function, Y=E[C]+1.5·σ(C)")


def part_B4_cashflow_vs_pnl(base_cfg):
    """Cash Flow vs Accounting P&L — Paper Sections 3.1-3.3."""
    logger.info("=" * 65)
    logger.info("PART B.4 — Cash Flow vs Accounting P&L (Sections 3.1-3.3)")
    logger.info("  Paper: 'the accounting P&L approach gives much better results'")
    logger.info("=" * 65)
    rows = []
    for Tl, T in [("1m", 1/12), ("3m", 3/12)]:
        ns = max(int(round(252*T)), 5)
        cfg = copy.deepcopy(base_cfg)
        cfg["simulation"]["maturity"] = T; cfg["simulation"]["n_steps"] = ns
        V0 = bs_price(cfg)
        logger.info(f"  ► Accounting P&L, T={Tl}")
        r_pnl = run(cfg, ProcessType.GBM, AgentType.DeepDPG, BenchmarkType.BsDelta)
        logger.info(f"  ► Cash Flow, T={Tl}")
        r_cf = run_cashflow(cfg, ProcessType.GBM, AgentType.DeepDPG)
        for form, m, cv in [("Accounting P&L", r_pnl["agent"], r_pnl["cv"]),
                            ("Cash Flow",       r_cf["metrics"], r_cf["cv"])]:
            rows.append({"T": Tl, "Formulation": form,
                "Mean%": m["mean"]/V0*100, "Std%": m["std"]/V0*100,
                "Y(0)%": m["Y"]/V0*100,
                "Y 95%CI": f"[{m['Y_lo']/V0*100:.1f}, {m['Y_hi']/V0*100:.1f}]",
                "conv": "✅" if cv["ok"] else "⚠️"})
    return ptable(rows, "B.4 — Cash Flow vs Accounting P&L\n"
                        "  DeepDPG, GBM, daily, κ=1%, S₀=100, σ=20%\n"
                        "  'Temporal credit assignment problem' (Minsky, 1961)")


# ======================================================================
#  PLOTS
# ======================================================================

def _sv(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  📊 {FIG/name}")


def plot_training(tr, title, fn="training_curve.png"):
    c = [sum(e.costs) for e in tr.episodes["train"]]
    w = max(len(c)//50, 10)
    sm = pd.Series(c).rolling(w, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(c, alpha=.12, color="steelblue")
    ax.plot(sm, color="steelblue", lw=2, label=f"Rolling avg (w={w})")
    ax.set(xlabel="Episode", ylabel="Hedging cost", title=title)
    ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout(); _sv(fig, fn)


def plot_convergence(tr, cv, title, fn="convergence.png"):
    """Training curve with convergence windows highlighted."""
    c = [sum(e.costs) for e in tr.episodes["train"]]
    n = len(c); w = cv["w"]
    sm = pd.Series(c).rolling(w, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(c, alpha=.08, color="steelblue")
    ax.plot(sm, color="steelblue", lw=2, label=f"Rolling avg (w={w})")
    ax.axvspan(n-2*w, n-w, alpha=.15, color="orange", label="Previous window")
    ax.axvspan(n-w, n,     alpha=.15, color="green",  label="Last window")
    ax.axhline(cv["prev"], color="orange", ls="--", lw=1.5)
    ax.axhline(cv["last"], color="green",  ls="--", lw=1.5)
    status = "CONVERGED" if cv["ok"] else "NOT CONVERGED"
    ax.set_title(f"{title}\n{status} (Δ={cv['rc']:.2%}, threshold=5%)", fontsize=10)
    ax.set(xlabel="Episode", ylabel="Hedging cost")
    ax.legend(loc="upper right"); ax.grid(True, alpha=.3)
    fig.tight_layout(); _sv(fig, fn)


def plot_costs(ac, bc, title, fn="cost_dist.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    lo, hi = min(min(ac), min(bc)), max(max(ac), max(bc))
    bins = np.linspace(lo, hi, 60)
    ax.hist(bc, bins=bins, alpha=.5, label="Delta", color="salmon", edgecolor="white")
    ax.hist(ac, bins=bins, alpha=.5, label="RL",    color="steelblue", edgecolor="white")
    ax.axvline(np.mean(bc), color="salmon",   ls="--", lw=2,
               label=f"Δ mean={np.mean(bc):.4f}")
    ax.axvline(np.mean(ac), color="steelblue", ls="--", lw=2,
               label=f"RL mean={np.mean(ac):.4f}")
    ax.set(xlabel="Total hedging cost", ylabel="Frequency", title=title)
    ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout(); _sv(fig, fn)


def plot_scatter(ea, eb, title, fn="scatter.png"):
    """Exhibit 5: RL position vs BS delta position."""
    ag, dl = [], []
    for a, b in zip(ea.episodes["eval_agent"], eb.episodes["eval_benchmark"]):
        ag.extend(a.actions); dl.extend(b.actions)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(dl, ag, alpha=.02, s=3, color="steelblue")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=1, label="H_RL = Δ_BS")
    ax.fill_between(lims, lims, [lims[1]]*2, alpha=.04, color="green",
                    label="Over-hedged")
    ax.fill_between(lims, [lims[0]]*2, lims, alpha=.04, color="red",
                    label="Under-hedged")
    ax.set(xlabel="BS Delta position", ylabel="RL position", title=title)
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=.3)
    fig.tight_layout(); _sv(fig, fn)


def plot_freq_bars(df, title, fn="freq_bars.png"):
    """Y(0) + improvement across rebalancing frequencies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df)); w = .35; lam = 1.5
    # Left: Y(0)
    ax = axes[0]
    ax.bar(x-w/2, df["Δ Mean%"]+lam*df["Δ Std%"], w,
           label="Delta Y(0)", color="salmon", edgecolor="white")
    ax.bar(x+w/2, df["RL Mean%"]+lam*df["RL Std%"], w,
           label="RL Y(0)", color="steelblue", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["Rebal"])
    ax.set(ylabel="Y(0) as % of V₀"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=.3, axis="y")
    # Right: improvement
    ax = axes[1]
    cols = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["Y improv%"]]
    ax.bar(x, df["Y improv%"], .5, color=cols, edgecolor="white")
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(df["Rebal"])
    ax.set(ylabel="Y(0) improvement %",
           title="RL improvement vs Delta hedging")
    ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout(); _sv(fig, fn)


def plot_sabr_benchmarks(df, title, fn="sabr_bench.png"):
    """3 bars: Practitioner Δ, Bartlett Δ, RL across frequencies."""
    x = np.arange(len(df)); w = .25; lam = 1.5
    rl_y = df["RL Mean%"] + lam * df["RL Std%"]
    pr_y = df["Practitioner Δ Mean%"] + lam * df["Practitioner Δ Std%"]
    ba_y = df["Bartlett Δ Mean%"] + lam * df["Bartlett Δ Std%"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x-w, pr_y, w, label="Practitioner Δ", color="salmon", edgecolor="white")
    ax.bar(x,   ba_y, w, label="Bartlett Δ",     color="#f39c12", edgecolor="white")
    ax.bar(x+w, rl_y, w, label="RL (DeepDPG)",   color="steelblue", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["Rebal"])
    ax.set(ylabel="Y(0) as % of V₀", title=title)
    ax.legend(); ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout(); _sv(fig, fn)


def plot_process_comparison(df, title, fn="process_comp.png"):
    """RL vs Delta across processes and maturities."""
    lam = 1.5
    labels = [f"{t}—{p}" for t, p in zip(df["T"], df["Process"])]
    x = np.arange(len(df)); w = .35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x-w/2, df["Δ Mean%"]+lam*df["Δ Std%"], w,
           label="Delta Y(0)", color="salmon", edgecolor="white")
    ax.bar(x+w/2, df["RL Mean%"]+lam*df["RL Std%"], w,
           label="RL Y(0)", color="steelblue", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set(ylabel="Y(0) % of V₀", title=title)
    ax.legend(); ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout(); _sv(fig, fn)


def plot_maturity_comparison(df, title, fn="maturity_comp.png"):
    """Mean + Std + improvement across maturities."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(df)); w = .35
    for ax, cols, sub in zip(axes[:2],
        [("Δ Mean%","RL Mean%"), ("Δ Std%","RL Std%")],
        ["Mean cost (% of V₀)", "Std cost (% of V₀)"]):
        ax.bar(x-w/2, df[cols[0]], w, label="Delta", color="salmon", edgecolor="white")
        ax.bar(x+w/2, df[cols[1]], w, label="RL", color="steelblue", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(df["Maturity"])
        ax.set(ylabel="% of V₀", title=f"{title}\n{sub}")
        ax.legend(); ax.grid(True, alpha=.3, axis="y")
    ax = axes[2]
    cols_c = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["Y improv%"]]
    ax.bar(x, df["Y improv%"], .5, color=cols_c, edgecolor="white")
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(df["Maturity"])
    ax.set(ylabel="Y(0) improv %", title="RL improvement vs Delta")
    ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout(); _sv(fig, fn)


def plot_agents(df, title, fn="agents.png"):
    """Y(0) per agent with CI."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df))
    colors = ["#e74c3c", "#f39c12", "#2980b9"][:len(df)]
    ax = axes[0]
    ax.bar(x, df["Y(0)%"], .45, color=colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df["Agent"])
    ax.set(ylabel="Y(0) % of V₀", title=title)
    ax.grid(True, alpha=.3, axis="y")
    ax = axes[1]
    imp = df["Y vs Δ%"]
    bar_c = ["#2ecc71" if v > 0 else "#e74c3c" for v in imp]
    ax.bar(x, imp, .45, color=bar_c, edgecolor="white")
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(df["Agent"])
    ax.set(ylabel="Y(0) improv vs Delta %",
           title="Improvement over BS Delta\n"
                 "DQN/DoubleDQN: 21-grid | DeepDPG: continuous")
    ax.grid(True, alpha=.3, axis="y")
    fig.tight_layout(); _sv(fig, fn)


def plot_cashflow_vs_pnl(df, fn="B4_cf_vs_pnl.png"):
    """Side-by-side: Accounting P&L vs Cash Flow for each T."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lam = 1.5
    for ax, Tl in zip(axes, ["1m", "3m"]):
        sub = df[df["T"] == Tl]
        x = np.arange(len(sub))
        y_vals = sub["Mean%"].values + lam * sub["Std%"].values
        colors = ["steelblue", "salmon"]
        bars = ax.bar(x, y_vals, .5, color=colors, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(sub["Formulation"].values)
        for bar, val in zip(bars, y_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set(ylabel="Y(0) = E[C]+λσ(C) % of V₀",
               title=f"T={Tl} — Cash Flow vs Accounting P&L\n"
                     f"DeepDPG, GBM, daily, κ=1%")
        ax.grid(True, alpha=.3, axis="y")
    fig.suptitle("Paper finding: Accounting P&L ≫ Cash Flow\n"
                 "(temporal credit assignment problem, Minsky 1961)",
                 fontsize=11, y=1.03)
    fig.tight_layout(); _sv(fig, fn)


# ======================================================================
#  MAIN
# ======================================================================

def main():
    logger.info("╔═══════════════════════════════════════════════════════════════╗")
    logger.info("║  Deep Hedging — Cao, Chen, Hull & Poulos (2021)              ║")
    logger.info("║  PART A: Paper Replication  |  PART B: Extensions            ║")
    logger.info("╚═══════════════════════════════════════════════════════════════╝\n")

    cfg = json_to_dict("config.json")

    # ══════════════════════════════════════════════════════════════════
    #   PART A — PAPER REPLICATION
    # ══════════════════════════════════════════════════════════════════

    # A.1 — GBM, T=1 month (Exhibit 3)
    df_a1 = part_A1_A2(cfg, T=1/12, exhibit_name="Exhibit 3")
    plot_freq_bars(df_a1,
        "Exhibit 3 — GBM, T=1m, Short ATM Call\n"
        "S₀=100, K=100, σ=20%, μ=5%, r=0, q=0, κ=1%",
        "A1_exhibit3_gbm_1m.png")

    # A.2 — GBM, T=3 months (Exhibit 4)
    df_a2 = part_A1_A2(cfg, T=3/12, exhibit_name="Exhibit 4")
    plot_freq_bars(df_a2,
        "Exhibit 4 — GBM, T=3m, Short ATM Call\n"
        "S₀=100, K=100, σ=20%, μ=5%, r=0, q=0, κ=1%",
        "A2_exhibit4_gbm_3m.png")

    # A.3 — SABR, T=1 month (Exhibit 6)
    df_a3 = part_A3_A4(cfg, T=1/12, exhibit_name="Exhibit 6")
    plot_sabr_benchmarks(df_a3,
        "Exhibit 6 — SABR β=1, T=1m, Short ATM Call\n"
        "Practitioner Δ vs Bartlett Δ vs RL (DeepDPG)\n"
        "σ₀=20%, ρ=-0.4, ν=0.6, κ=1%",
        "A3_exhibit6_sabr_1m.png")

    # A.4 — SABR, T=3 months (Exhibit 7)
    df_a4 = part_A3_A4(cfg, T=3/12, exhibit_name="Exhibit 7")
    plot_sabr_benchmarks(df_a4,
        "Exhibit 7 — SABR β=1, T=3m, Short ATM Call\n"
        "Practitioner Δ vs Bartlett Δ vs RL (DeepDPG)\n"
        "σ₀=20%, ρ=-0.4, ν=0.6, κ=1%",
        "A4_exhibit7_sabr_3m.png")

    # A.5 — Baseline plots: training, convergence, costs, scatter
    logger.info("\n--- Generating baseline plots (Exhibit 5 etc.) ---")
    cfg_1m = copy.deepcopy(cfg)
    cfg_1m["simulation"]["maturity"] = 1/12
    cfg_1m["simulation"]["n_steps"] = 21
    r = run(cfg_1m, ProcessType.GBM, AgentType.DeepDPG, BenchmarkType.BsDelta)

    plot_training(r["tr"],
        "Training curve — DeepDPG + PER + ε-greedy, GBM, T=1m, daily\n"
        "Short ATM Call, S₀=100, σ=20%, κ=1%, Y=E[C]+1.5·σ(C)",
        "A5_training_curve.png")

    plot_convergence(r["tr"], r["cv"],
        "Convergence check — DeepDPG, GBM, T=1m, daily\n"
        "S₀=100, σ=20%, κ=1%",
        "A5_convergence.png")

    plot_costs(r["ac"], r["bc"],
        "Cost distribution — DeepDPG vs BS Delta\n"
        "GBM, T=1m, daily rebal, S₀=100, σ=20%, κ=1%\n"
        f"Welch p={r['tt']['p']:.2e} ({sig_stars(r['tt']['p'])})",
        "A5_cost_distribution.png")

    plot_scatter(r["ea"], r["eb"],
        "Over/Under-hedging — DeepDPG vs BS Delta (Exhibit 5)\n"
        "GBM, T=1m, daily, S₀=100, K=100, σ=20%, κ=1%",
        "A5_over_under_hedge.png")

    # ══════════════════════════════════════════════════════════════════
    #   PART B — EXTENSIONS
    # ══════════════════════════════════════════════════════════════════

    # B.1 — SVJ process
    df_b1 = part_B1_svj(cfg)
    plot_process_comparison(df_b1,
        "B.1 — Process comparison: GBM vs SVJ\n"
        "DeepDPG vs BS Delta, daily rebal, κ=1%, S₀=100",
        "B1_process_comparison.png")

    # B.2 — Longer maturities
    df_b2 = part_B2_maturities(cfg)
    plot_maturity_comparison(df_b2,
        "B.2 — Maturity comparison\n"
        "GBM, DeepDPG, daily, κ=1%",
        "B2_maturity_comparison.png")

    # B.3 — Agent comparison
    df_b3 = part_B3_agents(cfg)
    plot_agents(df_b3,
        "B.3 — Agent comparison, GBM T=3m, daily, κ=1%\n"
        "All: dual Q-function, Y=E[C]+1.5·σ(C)",
        "B3_agent_comparison.png")

    # B.4 — Cash Flow vs Accounting P&L
    df_b4 = part_B4_cashflow_vs_pnl(cfg)
    plot_cashflow_vs_pnl(df_b4, "B4_cashflow_vs_pnl.png")

    logger.info("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
