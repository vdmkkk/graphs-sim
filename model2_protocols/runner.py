"""Experiment orchestration for Model 2.

Builds the physical network once per (scenario, seed), then runs every
protocol on that *same* network so the only variable is the routing logic.
Produces a tidy results table plus the comparison artefacts the NIR needs:
PDR / latency / throughput / overhead / energy per protocol and scenario,
and the latency-vs-throughput trade-off from a load sweep.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import SimConfig
from .metrics import Metrics
from .network.deployment import build_network, reliable_range_m
from .protocols import ALL_PROTOCOLS
from .scenarios import Scenario, SCENARIOS, load_sweep_rates, baseline_scenario, TrafficModel
from .sim import Simulator


class ProtocolExperiment:
    def __init__(self, cfg: SimConfig | None = None, protocols=None,
                 reps: int = 3, num_nodes: int = 40, drain_s: float = 120.0):
        self.cfg = cfg or SimConfig()
        self.protocols = protocols or ALL_PROTOCOLS
        self.reps = reps
        self.num_nodes = num_nodes
        self.drain_s = drain_s

    # ------------------------------------------------------------------
    def run_single(self, protocol_cls, scenario: Scenario, seed: int,
                   cfg: SimConfig | None = None) -> dict:
        cfg = cfg or self.cfg
        network = build_network(cfg, scenario.num_nodes, scenario.deployment,
                                rng=random.Random(seed))
        metrics = Metrics(cfg)
        protocol = protocol_cls(cfg)
        sim = Simulator(network, cfg, protocol, metrics, random.Random(seed + 1))

        traffic = scenario.traffic.generate(network, random.Random(seed + 2),
                                             cfg.sim_time, cfg.data_payload_bytes)
        dynamics = scenario.node_dynamics(network, random.Random(seed + 3), cfg.sim_time)

        for t, node_id, alive in dynamics:
            sim.schedule_at(t, self._set_alive, network, node_id, alive)
        for t, message in traffic:
            sim.schedule_at(t, sim.app_send, message)

        sim.run(until=cfg.sim_time + self.drain_s)

        row = {
            "protocol": protocol.name,
            "proactive": protocol.proactive,
            "scenario": scenario.name,
            "deployment": scenario.deployment.value,
            "sf": cfg.phy.sf,
            "seed": seed,
            "num_nodes": network.n,
            "mean_degree": round(network.mean_degree(), 2),
            "reliable_range_m": round(reliable_range_m(cfg), 1),
            "offered_rate_per_min": scenario.traffic.rate_per_min,
        }
        row.update(metrics.summarize(network, sim.channel, cfg.sim_time))
        return row

    @staticmethod
    def _set_alive(network, node_id: int, alive: bool) -> None:
        network.nodes[node_id].alive = alive

    # ------------------------------------------------------------------
    def run_matrix(self, scenario_names=("baseline", "churn", "partition")) -> pd.DataFrame:
        rows = []
        combos = [(s, p, r) for s in scenario_names
                  for p in self.protocols for r in range(self.reps)]
        for sname, proto, rep in tqdm(combos, desc="protocol x scenario", unit="run"):
            scenario = SCENARIOS[sname](num_nodes=self.num_nodes)
            rows.append(self.run_single(proto, scenario, seed=self.cfg.seed + rep))
        return pd.DataFrame(rows)

    def run_load_sweep(self) -> pd.DataFrame:
        rows = []
        rates = load_sweep_rates()
        combos = [(rate, p, r) for rate in rates
                  for p in self.protocols for r in range(max(1, self.reps - 1))]
        for rate, proto, rep in tqdm(combos, desc="load sweep", unit="run"):
            scenario = baseline_scenario(num_nodes=self.num_nodes)
            scenario.name = "load_sweep"
            scenario.traffic = TrafficModel(rate_per_min=rate)
            row = self.run_single(proto, scenario, seed=self.cfg.seed + rep)
            rows.append(row)
        return pd.DataFrame(rows)

    def run_deployment_sweep(self, deployments=None, reps: int | None = None
                             ) -> pd.DataFrame:
        """Baseline scenario across geographic layouts (uniform/clustered/...)."""
        from .network.deployment import DeploymentScenario
        deployments = deployments or list(DeploymentScenario)
        reps = reps or self.reps
        rows = []
        combos = [(dep, p, r) for dep in deployments
                  for p in self.protocols for r in range(reps)]
        for dep, proto, rep in tqdm(combos, desc="deployment sweep", unit="run"):
            scenario = baseline_scenario(num_nodes=self.num_nodes, deployment=dep)
            scenario.name = "deployment"
            rows.append(self.run_single(proto, scenario, seed=self.cfg.seed + rep))
        return pd.DataFrame(rows)

    def run_node_sweep(self, node_counts=None, reps: int | None = None
                       ) -> pd.DataFrame:
        """Baseline scenario across network sizes (scalability)."""
        node_counts = node_counts or [15, 25, 40, 60, 80]
        reps = reps or max(3, self.reps // 2)
        rows = []
        combos = [(n, p, r) for n in node_counts
                  for p in self.protocols for r in range(reps)]
        for n, proto, rep in tqdm(combos, desc="node sweep", unit="run"):
            scenario = baseline_scenario(num_nodes=n)
            scenario.name = "node_sweep"
            rows.append(self.run_single(proto, scenario, seed=self.cfg.seed + rep))
        return pd.DataFrame(rows)

    def run_sf_sweep(self, sf_list=None, reps: int | None = None) -> pd.DataFrame:
        """Baseline scenario across spreading factors (SF7..SF12)."""
        from .config import LoRaPHY
        sf_list = sf_list or [7, 8, 9, 10, 11, 12]
        reps = reps or max(3, self.reps // 2)
        rows = []
        combos = [(sf, p, r) for sf in sf_list
                  for p in self.protocols for r in range(reps)]
        for sf, proto, rep in tqdm(combos, desc="SF sweep", unit="run"):
            cfg = self.cfg.clone(phy=LoRaPHY(sf=sf))
            scenario = baseline_scenario(num_nodes=self.num_nodes)
            scenario.name = "sf_sweep"
            rows.append(self.run_single(proto, scenario, seed=self.cfg.seed + rep,
                                        cfg=cfg))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    @staticmethod
    def _agg_ci(df: pd.DataFrame, group_cols: list[str], metrics: list[str]
                ) -> pd.DataFrame:
        """Mean, std and 95% confidence half-width for each metric per group."""
        import math
        g = df.groupby(group_cols)
        out = g[metrics].mean().round(4)
        for m in metrics:
            std = g[m].std()
            n = g[m].count()
            ci = 1.96 * std / n.pow(0.5)
            out[f"{m}_std"] = std.round(4)
            out[f"{m}_ci95"] = ci.round(4)
        out["n"] = g.size()
        return out

    def export_analytics(self, matrix: pd.DataFrame, sweep: pd.DataFrame,
                         out_dir: str | Path,
                         deployment: pd.DataFrame | None = None,
                         node_sweep: pd.DataFrame | None = None,
                         sf_sweep: pd.DataFrame | None = None) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        matrix.to_csv(out_dir / "results_matrix.csv", index=False)
        sweep.to_csv(out_dir / "results_load_sweep.csv", index=False)

        agg_cols = {
            "pdr": "mean", "latency_mean_s": "mean", "latency_p95_s": "mean",
            "goodput_bps": "mean", "control_airtime_frac": "mean",
            "tx_per_delivery": "mean", "channel_utilisation": "mean",
            "collision_rate": "mean", "energy_per_bit_mj": "mean",
            "mean_node_lifetime_h": "mean", "drops_duty_cycle": "mean",
        }
        by_proto_scen = (matrix.groupby(["scenario", "protocol"])
                         .agg(agg_cols).round(4))
        by_proto_scen.to_csv(out_dir / "summary_by_protocol_scenario.csv")

        # Same table but with dispersion: mean, std and 95% CI half-width.
        ci_metrics = ["pdr", "latency_mean_s", "tx_per_delivery",
                      "control_airtime_frac", "energy_per_bit_mj",
                      "mean_node_lifetime_h"]
        self._agg_ci(matrix, ["scenario", "protocol"], ci_metrics).to_csv(
            out_dir / "summary_ci.csv")

        matrix.pivot_table(index="protocol", columns="scenario", values="pdr",
                           aggfunc="mean").round(3).to_csv(out_dir / "pdr_pivot.csv")
        matrix.pivot_table(index="protocol", columns="scenario",
                           values="latency_mean_s", aggfunc="mean").round(3).to_csv(
            out_dir / "latency_pivot.csv")

        # Resilience: PDR retention churn/partition relative to baseline.
        base = (matrix[matrix.scenario == "baseline"].groupby("protocol")["pdr"]
                .mean())
        res_rows = []
        for scen in ("churn", "partition"):
            sub = matrix[matrix.scenario == scen].groupby("protocol")["pdr"].mean()
            for proto in sub.index:
                b = base.get(proto, float("nan"))
                res_rows.append({
                    "protocol": proto, "scenario": scen,
                    "pdr": round(sub[proto], 3),
                    "baseline_pdr": round(b, 3) if b == b else None,
                    "retention": round(sub[proto] / b, 3) if b else None,
                })
        pd.DataFrame(res_rows).to_csv(out_dir / "resilience.csv", index=False)

        # Extended sweeps (optional).
        if deployment is not None and not deployment.empty:
            deployment.to_csv(out_dir / "results_deployment.csv", index=False)
            deployment.pivot_table(index="protocol", columns="deployment",
                                   values="pdr", aggfunc="mean").round(3).to_csv(
                out_dir / "pdr_by_deployment.csv")
        if node_sweep is not None and not node_sweep.empty:
            node_sweep.to_csv(out_dir / "results_node_sweep.csv", index=False)
            node_sweep.pivot_table(index="protocol", columns="num_nodes",
                                   values="pdr", aggfunc="mean").round(3).to_csv(
                out_dir / "pdr_by_nodes.csv")
        if sf_sweep is not None and not sf_sweep.empty:
            sf_sweep.to_csv(out_dir / "results_sf_sweep.csv", index=False)
            sf_sweep.pivot_table(index="protocol", columns="sf",
                                 values="pdr", aggfunc="mean").round(3).to_csv(
                out_dir / "pdr_by_sf.csv")

        self._make_plots(matrix, sweep, out_dir,
                         deployment=deployment, node_sweep=node_sweep,
                         sf_sweep=sf_sweep)

    def _make_plots(self, matrix: pd.DataFrame, sweep: pd.DataFrame,
                    out_dir: Path, deployment: pd.DataFrame | None = None,
                    node_sweep: pd.DataFrame | None = None,
                    sf_sweep: pd.DataFrame | None = None) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - plotting is optional
            return

        # 1) PDR vs offered load
        fig, ax = plt.subplots(figsize=(8, 5))
        for proto, grp in sweep.groupby("protocol"):
            g = grp.groupby("offered_rate_per_min")["pdr"].mean()
            ax.plot(g.index, g.values, marker="o", label=proto)
        ax.set_xlabel("Предлагаемая нагрузка, сообщений/мин")
        ax.set_ylabel("Коэффициент доставки (PDR)")
        ax.set_title("Зависимость PDR от предлагаемой нагрузки")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "pdr_vs_load.png", dpi=150)
        plt.close(fig)

        # 2) Latency vs goodput trade-off
        fig, ax = plt.subplots(figsize=(8, 5))
        for proto, grp in sweep.groupby("protocol"):
            g = grp.groupby("offered_rate_per_min").agg(
                {"goodput_bps": "mean", "latency_mean_s": "mean"})
            ax.plot(g["goodput_bps"], g["latency_mean_s"], marker="o", label=proto)
        ax.set_xlabel("Полезная пропускная способность, бит/с")
        ax.set_ylabel("Средняя сквозная задержка, с")
        ax.set_title("Компромисс «задержка — пропускная способность»")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "latency_vs_throughput.png", dpi=150)
        plt.close(fig)

        # 3) PDR by scenario (resilience bar chart) with 95% CI whiskers
        fig, ax = plt.subplots(figsize=(9, 5))
        piv = matrix.pivot_table(index="protocol", columns="scenario",
                                 values="pdr", aggfunc="mean")
        err = matrix.pivot_table(index="protocol", columns="scenario",
                                 values="pdr", aggfunc="sem") * 1.96
        piv.plot(kind="bar", ax=ax, yerr=err, capsize=3)
        ax.set_ylabel("Коэффициент доставки (PDR)")
        ax.set_title("Доставка по сценариям (устойчивость, 95 % ДИ)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "pdr_by_scenario.png", dpi=150)
        plt.close(fig)

        base = matrix[matrix.scenario == "baseline"]

        # 4) Energy per delivered bit, by protocol (baseline)
        fig, ax = plt.subplots(figsize=(8, 5))
        g = base.groupby("protocol")["energy_per_bit_mj"]
        m, e = g.mean(), g.sem() * 1.96
        ax.bar(m.index, m.values, yerr=e.values, capsize=3, color="#b5683b")
        ax.set_ylabel("Энергия на доставленный бит, мДж")
        ax.set_title("Энергоэффективность по протоколам (baseline, 95% CI)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "energy_per_bit.png", dpi=150)
        plt.close(fig)

        # 5) Mean node lifetime, by protocol (baseline)
        fig, ax = plt.subplots(figsize=(8, 5))
        g = base.groupby("protocol")["mean_node_lifetime_h"]
        m, e = g.mean(), g.sem() * 1.96
        ax.bar(m.index, m.values, yerr=e.values, capsize=3, color="#5a7da4")
        ax.set_ylabel("Средний срок жизни узла, ч")
        ax.set_title("Время жизни сети по протоколам (baseline, 95% CI)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "node_lifetime.png", dpi=150)
        plt.close(fig)

        # 6) Control-airtime overhead vs PDR (does overhead buy reliability?)
        fig, ax = plt.subplots(figsize=(8, 5))
        for proto, grp in base.groupby("protocol"):
            ax.scatter(grp["control_airtime_frac"] * 100, grp["pdr"],
                       label=proto, s=45, alpha=0.8)
        ax.set_xlabel("Доля служебного эфира, %")
        ax.set_ylabel("PDR")
        ax.set_title("Накладные расходы против надёжности (baseline)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "overhead_vs_pdr.png", dpi=150)
        plt.close(fig)

        # 7) PDR vs network size (scalability)
        if node_sweep is not None and not node_sweep.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for proto, grp in node_sweep.groupby("protocol"):
                gg = grp.groupby("num_nodes")["pdr"].mean()
                ax.plot(gg.index, gg.values, marker="o", label=proto)
            ax.set_xlabel("Число узлов")
            ax.set_ylabel("PDR")
            ax.set_title("Масштабируемость: PDR от размера сети")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "pdr_vs_nodes.png", dpi=150)
            plt.close(fig)

        # 8) PDR vs spreading factor
        if sf_sweep is not None and not sf_sweep.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for proto, grp in sf_sweep.groupby("protocol"):
                gg = grp.groupby("sf")["pdr"].mean()
                ax.plot(gg.index, gg.values, marker="o", label=proto)
            ax.set_xlabel("Spreading factor (SF)")
            ax.set_ylabel("PDR")
            ax.set_title("Влияние SF на доставку (baseline)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "pdr_vs_sf.png", dpi=150)
            plt.close(fig)

        # 9) PDR by deployment layout
        if deployment is not None and not deployment.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            piv = deployment.pivot_table(index="protocol", columns="deployment",
                                         values="pdr", aggfunc="mean")
            err = deployment.pivot_table(index="protocol", columns="deployment",
                                         values="pdr", aggfunc="sem") * 1.96
            piv.plot(kind="bar", ax=ax, yerr=err, capsize=3)
            ax.set_ylabel("PDR")
            ax.set_title("Доставка по типам размещения (95% CI)")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(out_dir / "pdr_by_deployment.png", dpi=150)
            plt.close(fig)

    # ------------------------------------------------------------------
    @staticmethod
    def print_summary(matrix: pd.DataFrame) -> None:
        print("\n" + "=" * 78)
        print("MODEL 2 - PROTOCOL SUITABILITY SUMMARY")
        print("=" * 78)
        cols = ["pdr", "latency_mean_s", "goodput_bps", "control_airtime_frac",
                "tx_per_delivery", "collision_rate", "energy_per_bit_mj",
                "mean_node_lifetime_h"]
        summary = (matrix.groupby(["scenario", "protocol"])[cols]
                   .mean().round(3))
        print(summary.to_string())
