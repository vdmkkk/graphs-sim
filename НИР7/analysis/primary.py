"""Primary calculation of hypotheses Г1–Г6 (НИР7): newest finished full ONEmesh snapshot + all MQTT.

    python -m analysis.primary              # -> results/<snapshot ts>/ (tables, figures, summary.md)

Every link observation is "tx was heard directly by rx with this SNR (and RSSI for MQTT)":
traceroute hops, NeighborInfo entries, MQTT last hops (zero-hop senders and resolved relays).
"""
import json
import sys
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.linalg import lsqr  # noqa: E402

from . import common as c  # noqa: E402

RNG = np.random.default_rng(7)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NIR6 = dict(n=2.08, sigma=3.57, pl0=127.41, d0_km=0.040)  # Bor et al. 2016, used in НИР6
SNR_MIN_SF9 = -12.5   # demodulation limit, SF9 (SX1262 datasheet)
NF_DB = 6.0           # typical SX1262 receiver noise figure
P_TX_RU = 20.0        # dBm, region RU limit in Meshtastic firmware
BASE_MODEL = "HELTEC_V3"
SNR_TRUNC = (-16.0, 10.0)  # observable SNR window: below - not decoded, above - SX126x estimate saturates
NOISE_SNR_MAX = -5.0  # dB: noise floor only from packets where RSSI is noise-dominated and SNR not saturated


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2,
    })


# ---------------------------------------------------------------- statistics helpers

def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return (n + 1 - 2 * (np.cumsum(x) / x.sum()).sum()) / n


def top_share(x, frac=0.1):
    x = np.sort(np.asarray(x, float))[::-1]
    return x[:max(1, int(round(frac * len(x))))].sum() / x.sum()


def lorenz(x):
    x = np.sort(np.asarray(x, float))
    return np.r_[0, np.arange(1, len(x) + 1) / len(x)], np.r_[0, np.cumsum(x) / x.sum()]


def boot_ci(x, f=np.median, b=2000):
    x = np.asarray(x, float)
    if len(x) < 3:
        return (np.nan, np.nan)
    s = [f(RNG.choice(x, len(x))) for _ in range(b)]
    return tuple(np.percentile(s, [2.5, 97.5]))


def ols_2way(formula, df, g1="tx", g2="rx"):
    """OLS with standard errors clustered two-way (by transmitter and by receiver node)."""
    m = smf.ols(formula, df).fit()
    d = df.loc[m.model.data.row_labels]
    groups = np.column_stack([pd.factorize(d[g1])[0], pd.factorize(d[g2])[0]])
    return smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": groups})


def drop_singletons(df, g1, g2):
    while True:
        keep = (df[g1].map(df[g1].value_counts()) > 1) & (df[g2].map(df[g2].value_counts()) > 1)
        if keep.all():
            return df
        df = df[keep]


def within_2way(df, cols, g1, g2, tol=1e-9, maxit=1000):
    """Remove two sets of fixed effects (alternating projections)."""
    X = df[cols].to_numpy(float).copy()
    codes = [pd.factorize(df[g])[0] for g in (g1, g2)]
    counts = [np.bincount(k) for k in codes]
    for _ in range(maxit):
        prev = X.copy()
        for k, cnt in zip(codes, counts):
            sums = np.zeros((cnt.size, X.shape[1]))
            np.add.at(sums, k, X)
            X -= (sums / cnt[:, None])[k]
        if np.abs(X - prev).max() < tol:
            break
    return X, codes


def fe_slope(df, y, x, g1="tx", g2="rx"):
    """Slope of y on x with tx and rx fixed effects; two-way clustered SE; residual sigma."""
    d = drop_singletons(df[[y, x, g1, g2]].dropna(), g1, g2)
    if len(d) < 30:
        return None
    X, codes = within_2way(d, [y, x], g1, g2)
    fit = sm.OLS(X[:, 0], X[:, 1:]).fit(cov_type="cluster", cov_kwds={"groups": np.column_stack(codes)})
    dof = len(d) - 1 - (codes[0].max() + 1) - (codes[1].max() + 1) + 1
    sigma = np.sqrt((fit.resid ** 2).sum() / max(dof, 1))
    return dict(b=fit.params[0], se=fit.bse[0], n_links=len(d), n_tx=codes[0].max() + 1,
                n_rx=codes[1].max() + 1, sigma=sigma)


def coef_table(fit, prefix=""):
    ci = fit.conf_int()
    return pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "se": fit.bse.values,
                         "ci_lo": ci[0].values, "ci_hi": ci[1].values, "p": fit.pvalues.values,
                         "model": prefix})


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def fmt_ci(lo, hi, nd=1):
    return f"[{fmt(lo, nd)}; {fmt(hi, nd)}]"


# ---------------------------------------------------------------- data

def build(full):
    nodes = c.load_nodes(full)
    tr, routes = c.traceroute_links(full)
    nb = c.neighbour_links(full)
    mq = c.load_mqtt()
    zero = mq[(mq["hops"] == 0) & ~mq["self"]]
    pairs = pd.concat([tr[["tx", "rx"]], nb[["tx", "rx"]],
                       pd.DataFrame({"tx": zero["from"], "rx": zero["gw"]})])
    neigh = {}
    for a, b in pairs.itertuples(index=False):
        neigh.setdefault(int(b), set()).add(int(a))
        neigh.setdefault(int(a), set()).add(int(b))
    ml = c.mqtt_links(mq, neigh)
    obs = pd.concat([tr, nb, ml], ignore_index=True)
    obs["tx"], obs["rx"] = obs["tx"].astype("int64"), obs["rx"].astype("int64")
    return dict(full=full, nodes=nodes, tr=tr, nb=nb, mq=mq, ml=ml, obs=obs, routes=routes, neigh=neigh)


def overview(D):
    nodes, mq, obs = D["nodes"], D["mq"], D["obs"]
    zero = mq[(mq["hops"] == 0) & (mq["relay_node"] > 0)]
    relay_ok = (zero["relay_node"] == (zero["from"] & 0xFF)).mean() if len(zero) else np.nan
    multi = mq[mq["radio_ok"] & (mq["hops"] > 0)]
    resolved = D["ml"]["how"].eq("relay").sum() / max(len(multi), 1)
    t0, t1 = mq["t"].min(), mq["t"].max()
    return {
        "snapshot": D["full"].name,
        "nodes": len(nodes), "nodes_coords": int(nodes["lat"].notna().sum()),
        "nodes_hag": int(nodes["hag_m"].notna().sum()), "nodes_mobile": int(nodes["mobile"].sum()),
        "traceroute_obs": len(D["tr"]), "traceroutes_unique": len({r["packet_id"] for r in D["routes"]}),
        "neighbour_obs": len(D["nb"]), "mqtt_records": len(mq),
        "mqtt_from": datetime.fromtimestamp(t0, timezone.utc).strftime("%d.%m %H:%M"),
        "mqtt_to": datetime.fromtimestamp(t1, timezone.utc).strftime("%d.%m %H:%M"),
        "mqtt_hours": (t1 - t0) / 3600, "gateways": mq["gw"].nunique(),
        "mqtt_link_obs": len(D["ml"]), "mqtt_zero_hop": int((D["ml"]["how"] == "zero_hop").sum()),
        "relay_byte_matches_sender": relay_ok, "relay_resolved_share": resolved,
        "directed_links": obs.groupby(["tx", "rx"]).ngroups,
        "obs_by_source": obs["source"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------- Г1 + Г3

def h1_h3(D, out):
    L = c.link_table(D["obs"], D["nodes"])
    L = L[L["dist_ok"]].copy()
    L["lg_d"] = np.log10(L["d_km"])
    for s in ("tx", "rx"):
        L[f"lg_h_{s}"] = np.log10(L[f"{s}_hag_m"].clip(lower=2))
    T = f"Treatment('{BASE_MODEL}')"
    res = {"n_links_dist": len(L)}

    # Hardware: all links with good geometry (no heights needed).
    fa = ols_2way(f"snr ~ lg_d + C(tx_model, {T}) + C(rx_model, {T})", L)
    fb = ols_2way("snr ~ lg_d + C(tx_hw_class, Treatment('board')) + C(rx_hw_class, Treatment('board'))", L)
    # Heights: both ends known.
    H = L.dropna(subset=["lg_h_tx", "lg_h_rx"]).copy()
    res["n_links_height"] = len(H)
    res["n_nodes_height"] = len(set(H["tx"]) | set(H["rx"]))
    tables = [coef_table(fa, "hw_models"), coef_table(fb, "hw_classes")]
    if len(H) >= 40:
        full_f = f"snr ~ lg_d + lg_h_tx + lg_h_rx + C(tx_model, {T}) + C(rx_model, {T})"
        fh = ols_2way(full_f, H)
        r2_full = fh.rsquared
        r2_noh = smf.ols(f"snr ~ lg_d + C(tx_model, {T}) + C(rx_model, {T})", H).fit().rsquared
        r2_nohw = smf.ols("snr ~ lg_d + lg_h_tx + lg_h_rx", H).fit().rsquared
        q25, q75 = np.log10(np.percentile(H[["tx_hag_m", "rx_hag_m"]].clip(lower=2).values.ravel(), [25, 75]))
        hw_tx = [0.0] + [v for k, v in fh.params.items() if k.startswith("C(tx_model")]
        res.update(
            h_b_tx=fh.params["lg_h_tx"], h_se_tx=fh.bse["lg_h_tx"], h_p_tx=fh.pvalues["lg_h_tx"],
            h_b_rx=fh.params["lg_h_rx"], h_se_rx=fh.bse["lg_h_rx"], h_p_rx=fh.pvalues["lg_h_rx"],
            hag_q25=10 ** q25, hag_q75=10 ** q75,
            eff_height_iqr=(fh.params["lg_h_tx"] + fh.params["lg_h_rx"]) * (q75 - q25),
            eff_hw_range_tx=max(hw_tx) - min(hw_tx),
            partial_r2_height=r2_full - r2_noh, partial_r2_hw=r2_full - r2_nohw, r2_full=r2_full)
        tables.append(coef_table(fh, "height_both_ends"))
    # Heights, transmitter side only (larger sample).
    H1 = L.dropna(subset=["lg_h_tx"])
    if len(H1) >= 40:
        f1 = ols_2way(f"snr ~ lg_d + lg_h_tx + C(tx_model, {T}) + C(rx_model, {T})", H1)
        res.update(n_links_height_tx=len(H1), h1_b_tx=f1.params["lg_h_tx"], h1_se_tx=f1.bse["lg_h_tx"],
                   h1_p_tx=f1.pvalues["lg_h_tx"])
        tables.append(coef_table(f1, "height_tx_only"))
    pd.concat(tables).to_csv(out / "g1_g3_regressions.csv", index=False)

    # Г3: pocket vs board, both directions, plus the symmetric component of two-way links.
    for s in ("tx", "rx"):
        k = f"C({s}_hw_class, Treatment('board'))[T.pocket]"
        if k in fb.params:
            lo, hi = fb.conf_int().loc[k]
            res[f"pocket_{s}"], res[f"pocket_{s}_ci"], res[f"pocket_{s}_p"] = fb.params[k], (lo, hi), fb.pvalues[k]
    res["n_pocket_nodes_in_links"] = len(set(L.loc[L["tx_hw_class"] == "pocket", "tx"])
                                         | set(L.loc[L["rx_hw_class"] == "pocket", "rx"]))
    P = two_way_pairs(L)
    if len(P) >= 30:
        P["n_pocket"] = (P["a_cls"] == "pocket").astype(int) + (P["b_cls"] == "pocket").astype(int)
        P["n_other"] = (P["a_cls"] == "other").astype(int) + (P["b_cls"] == "other").astype(int)
        res.update(sym_n_pairs=len(P), sym_pairs_with_pocket=int((P["n_pocket"] > 0).sum()))
    if len(P) >= 30 and res["sym_pairs_with_pocket"] >= 5:
        fs = ols_2way("s ~ lg_d + n_pocket + n_other", P, "a", "b")
        lo, hi = fs.conf_int().loc["n_pocket"]
        res.update(sym_pocket=fs.params["n_pocket"], sym_pocket_ci=(lo, hi), sym_pocket_p=fs.pvalues["n_pocket"])

    ci = fa.conf_int()
    hw = []
    for sd in ("tx", "rx"):
        for m in c.MAIN_MODELS + ["OTHER"]:
            k = f"C({sd}_model, {T})[T.{m}]"
            if k in fa.params:
                hw.append((sd, m, fa.params[k], ci.loc[k, 0], ci.loc[k, 1]))
    res["hw_linklist"] = pd.DataFrame(hw, columns=["side", "model", "coef", "lo", "hi"])

    if "eff_height_iqr" in res:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        Hp = H.copy()
        Hp["resid"] = smf.ols(f"snr ~ lg_d + C(tx_model, {T}) + C(rx_model, {T})", Hp).fit().resid
        ax.scatter(Hp["tx_hag_m"].clip(lower=2), Hp["resid"], s=10, color=BLUE, alpha=0.45, lw=0,
                   label="линии (остаток SNR)")
        bins = np.logspace(np.log10(2), np.log10(max(Hp["tx_hag_m"].max(), 3)), 8)
        mid = pd.cut(Hp["tx_hag_m"].clip(lower=2), bins)
        med = Hp.groupby(mid, observed=True)["resid"].median()
        ax.plot([iv.mid for iv in med.index], med.values, "-o", color=ORANGE, ms=6, label="медиана по интервалам")
        ax.set_xscale("log")
        ax.set_xlabel("высота передатчика над поверхностью (SRTM), м")
        ax.set_ylabel("остаток SNR, дБ")
        ax.set_title("Г1. SNR линии и высота передатчика (после учёта расстояния и железа)", fontsize=10, loc="left")
        ax.legend(loc="lower right")
        fig.savefig(out / "fig_g1_height.png")
        plt.close(fig)
    return res, L


def plot_hardware(hw, out, source):
    """Hardware corrections to link SNR (tx and rx side) relative to BASE_MODEL, with 95 % CI."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    models = [m for m in c.MAIN_MODELS + ["OTHER"] if m != BASE_MODEL and (hw["model"] == m).any()]
    for j, (sd, col, mk, lab) in enumerate((("tx", BLUE, "o", "как передатчик"), ("rx", ORANGE, "s", "как приёмник"))):
        h = hw[hw["side"] == sd].set_index("model").reindex(models).dropna()
        xs = [models.index(m) + (j - 0.5) * 0.25 for m in h.index]
        ax.errorbar(xs, h["coef"], yerr=[h["coef"] - h["lo"], h["hi"] - h["coef"]], fmt=mk, color=col, ms=6,
                    capsize=3, lw=1.5, label=lab)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(models)), [c.SHORT[m] for m in models], rotation=20)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel(f"Δ SNR относительно {c.SHORT[BASE_MODEL]}, дБ")
    ax.set_title(f"Г1/Г3. Поправка модели устройства к SNR линии, 95 % ДИ ({source})", fontsize=10, loc="left")
    ax.legend(loc="lower left")
    fig.savefig(out / "fig_g3_hardware.png")
    plt.close(fig)


def two_way_pairs(L, a_col="tx", b_col="rx"):
    """Links observed in both directions: one row per unordered pair."""
    d = L[[a_col, b_col, "snr"]].rename(columns={a_col: "a", b_col: "b", "snr": "snr_ab"})
    r = L[[a_col, b_col, "snr"]].rename(columns={a_col: "b", b_col: "a", "snr": "snr_ba"})
    P = d.merge(r, on=["a", "b"])
    P = P[P["a"] < P["b"]].copy()
    P["delta"] = P["snr_ab"] - P["snr_ba"]
    P["s"] = (P["snr_ab"] + P["snr_ba"]) / 2
    extra = L.set_index([a_col, b_col])
    for col, src in (("lg_d", "lg_d"), ("a_cls", "tx_hw_class"), ("b_cls", "rx_hw_class")):
        if src in extra:
            P[col] = extra.loc[list(zip(P["a"], P["b"])), src].to_numpy()
    return P


# ---------------------------------------------------------------- Г2

def solve_potentials(P, nodes_idx=None, damp=0.0):
    """Least squares q: delta_ab = q_a - q_b; damp > 0 is ridge shrinkage of q towards 0.

    damp = 0: min-norm solution, zero mean per connected component.
    """
    ids = pd.Index(sorted(set(P["a"]) | set(P["b"]))) if nodes_idx is None else nodes_idx
    ia, ib = ids.get_indexer(P["a"]), ids.get_indexer(P["b"])
    m = len(P)
    A = coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[ia, ib])),
                   shape=(m, len(ids))).tocsr()
    q = lsqr(A, P["delta"].to_numpy(), damp=damp, atol=1e-10, btol=1e-10)[0]
    return pd.Series(q, index=ids)


def cv_potentials(P, damp, folds):
    """Out-of-sample prediction of asymmetry: fit q without a pair, predict that pair."""
    obs_cv, pred_cv = [], []
    for k in np.unique(folds):
        tr_, te = P[folds != k], P[folds == k]
        qk = solve_potentials(tr_, damp=damp)
        Gk = nx.Graph(list(zip(tr_["a"], tr_["b"])))
        cid = {n: i for i, cc in enumerate(nx.connected_components(Gk)) for n in cc}
        ok = te["a"].map(cid).notna() & (te["a"].map(cid) == te["b"].map(cid))
        te = te[ok]
        obs_cv.append(te["delta"].to_numpy())
        pred_cv.append(qk[te["a"]].to_numpy() - qk[te["b"]].to_numpy())
    obs_cv, pred_cv = np.concatenate(obs_cv), np.concatenate(pred_cv)
    return obs_cv, pred_cv, 1 - ((obs_cv - pred_cv) ** 2).sum() / (obs_cv ** 2).sum()


def h2(D, out, gw_noise):
    nodes = D["nodes"]
    links = D["obs"].groupby(["tx", "rx"])["snr"].median().reset_index()
    P = two_way_pairs(links)
    res = {"n_pairs": len(P)}
    if len(P) < 30:
        return res, None
    # Giant component of the pair graph.
    G = nx.Graph(list(zip(P["a"], P["b"])))
    comp = max(nx.connected_components(G), key=len)
    P = P[P["a"].isin(comp) & P["b"].isin(comp)].reset_index(drop=True)
    res.update(n_pairs_gc=len(P), n_nodes_gc=len(comp), delta_sd=P["delta"].std(),
               delta_abs_median=P["delta"].abs().median())
    q0 = solve_potentials(P)
    pred = q0[P["a"]].to_numpy() - q0[P["b"]].to_numpy()
    res["r2_in"] = 1 - ((P["delta"] - pred) ** 2).sum() / (P["delta"] ** 2).sum()
    # 5-fold cross-validation: does a node's potential predict asymmetry on pairs it was not fitted on?
    # Ridge damping is chosen by the same CV (slightly optimistic; the unregularized value is reported too).
    folds = np.random.default_rng(11).integers(0, 5, len(P))  # fixed folds: stable across runs
    cv = {d: cv_potentials(P, d, folds) for d in (0.0, 0.5, 1.0, 2.0, 4.0)}
    best = max(cv, key=lambda d: cv[d][2])
    obs_cv, pred_cv, r2 = cv[best]
    res.update(r2_cv=r2, r2_cv_nodamp=cv[0.0][2], damp=best, n_cv=len(obs_cv),
               r_cv=stats.pearsonr(obs_cv, pred_cv)[0])
    q = solve_potentials(P, damp=best)
    # Saturation check: drop pairs with SNR above +10 dB at either end.
    Ps = P[(P[["snr_ab", "snr_ba"]].max(axis=1) <= 10)]
    if len(Ps) > 30:
        res["r2_cv_unsat"] = cv_potentials(Ps.reset_index(drop=True), best, RNG.integers(0, 5, len(Ps)))[2]

    deg = pd.concat([P["a"], P["b"]]).value_counts()
    Q = pd.DataFrame({"q": q, "n_pairs": deg.reindex(q.index).to_numpy()})
    Q = Q.join(nodes[["model", "hw", "hw_class", "region_name", "role_name", "hag_m", "dist_center_km"]])
    Q["noise_dbm"] = gw_noise.reindex(Q.index)
    Qr = Q[Q["n_pairs"] >= 2]  # potentials resting on one pair are too noisy
    by = Qr.groupby("model")["q"].agg(["median", "mean", "count"])
    by["ci_lo"], by["ci_hi"] = zip(*[boot_ci(Qr.loc[Qr["model"] == m, "q"]) for m in by.index])
    by.to_csv(out / "g2_potential_by_model.csv")
    groups = [g["q"].to_numpy() for _, g in Qr.groupby("model") if len(g) >= 8]
    if len(groups) >= 2:
        res["kw_models_p"] = stats.kruskal(*groups).pvalue
    v4, v3 = Qr.loc[Qr["model"] == "HELTEC_V4", "q"], Qr.loc[Qr["model"] == "HELTEC_V3", "q"]
    if len(v4) >= 5 and len(v3) >= 5:
        res["v4_minus_v3"] = v4.median() - v3.median()
        bs = [np.median(RNG.choice(v4, len(v4))) - np.median(RNG.choice(v3, len(v3))) for _ in range(2000)]
        res["v4_minus_v3_ci"] = tuple(np.percentile(bs, [2.5, 97.5]))
        res["v4_v3_mw_p"] = stats.mannwhitneyu(v4, v3).pvalue
        res["n_v4"], res["n_v3"] = len(v4), len(v3)
    pk, bd = Qr.loc[Qr["hw_class"] == "pocket", "q"], Qr.loc[Qr["hw_class"] == "board", "q"]
    if len(pk) >= 5:
        res["pocket_minus_board_q"] = pk.median() - bd.median()
        res["pocket_board_mw_p"] = stats.mannwhitneyu(pk, bd).pvalue
        res["n_pocket_q"] = len(pk)
    qn = Qr.dropna(subset=["noise_dbm"])
    if len(qn) >= 8:
        res["q_noise_spearman"], res["q_noise_p"] = stats.spearmanr(qn["q"], qn["noise_dbm"])
        res["n_q_noise"] = len(qn)
    Qout = Q.reset_index(names="node").assign(node=lambda d: d["node"].map(c.anon))
    Qout.to_csv(out / "g2_node_potentials.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]
    ax.scatter(pred_cv, obs_cv, s=9, color=BLUE, alpha=0.4, lw=0)
    lim = np.percentile(np.abs(np.r_[obs_cv, pred_cv]), 99)
    ax.plot([-lim, lim], [-lim, lim], color=ORANGE, lw=1.5, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("прогноз q_A − q_B (вне выборки), дБ")
    ax.set_ylabel("SNR(A→B) − SNR(B→A), дБ")
    ax.set_title(f"а) асимметрия: R² вне выборки = {fmt(res['r2_cv'], 2)}", fontsize=10, loc="left")
    ax.legend(loc="upper left")
    ax = axs[1]
    order = [m for m in c.MAIN_MODELS + ["OTHER"] if (Qr["model"] == m).sum() >= 5]
    data = [Qr.loc[Qr["model"] == m, "q"] for m in order]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=ORANGE, lw=2), boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE),
                    whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
    ax.set_xticks(range(1, len(order) + 1), [f"{c.SHORT[m]}\n(n={len(d)})" for m, d in zip(order, data)],
                  fontsize=8)
    ax.set_ylabel("потенциал узла q = P + N, дБ")
    ax.set_title("б) потенциал по моделям (q, ≥2 пары на узел)", fontsize=10, loc="left")
    fig.savefig(out / "fig_g2_asymmetry.png")
    plt.close(fig)
    return res, Q


# ---------------------------------------------------------------- Г4

def truncreg(y, X, lo, hi):
    """Linear regression for a normal outcome observed only inside (lo, hi): ML estimate.

    Links weaker than the demodulation limit are never seen, and SX126x SNR saturates near +10 dB,
    so plain OLS flattens the distance slope (cf. censored path-loss fitting, Karttunen et al. 2017).
    """
    from scipy.optimize import minimize
    from scipy.stats import norm
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    s0 = np.log(np.std(y - X @ b0))

    def nll(th):
        b, sg = th[:-1], np.exp(th[-1])
        mu = X @ b
        pr = np.clip(norm.cdf((hi - mu) / sg) - norm.cdf((lo - mu) / sg), 1e-300, None)
        return -(norm.logpdf((y - mu) / sg) - th[-1] - np.log(pr)).sum()

    r = minimize(nll, np.r_[b0, s0], method="L-BFGS-B")
    return r.x[:-1], np.exp(r.x[-1])


def tobit2(y, left, right, X, lo, hi):
    """Two-limit Tobit: y* = Xb + e, e ~ N(0, s²); left-censored (y* <= lo), right-censored (y* >= hi)."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    u = ~left & ~right
    b0 = np.linalg.lstsq(X[u], y[u], rcond=None)[0]

    def nll(th):
        b, ls = th[:-1], th[-1]
        mu = X @ b
        sg = np.exp(ls)
        return -(norm.logpdf((y[u] - mu[u]) / sg).sum() - ls * u.sum()
                 + norm.logcdf((lo - mu[left]) / sg).sum() + norm.logsf((hi - mu[right]) / sg).sum())

    r = minimize(nll, np.r_[b0, np.log(np.std(y[u] - X[u] @ b0) + 1)], method="L-BFGS-B")
    return r.x[:-1], np.exp(r.x[-1])


def censored_pathloss(P, lo, hi, boot=100, extra=()):
    """Tobit of per-packet SNR on lg d + hardware of both ends; packets not received directly are
    left-censored at `lo`. Returns slope, its cluster-bootstrap SE (by sender), sigma, model inputs."""
    y = np.where(P["direct"], P["snr"], lo).astype(float)
    left = (~P["direct"].to_numpy()) | (y <= lo)
    right = P["direct"].to_numpy() & (y >= hi)
    cats = [BASE_MODEL] + [m for m in c.MAIN_MODELS + ["OTHER"] if m != BASE_MODEL]
    dm = pd.get_dummies(P[["tx_model", "rx_model"]].fillna("OTHER").astype(pd.CategoricalDtype(cats)),
                        drop_first=True, dtype=float)
    dm = dm.loc[:, dm.sum() >= 30]  # a model needs some packets to get its own coefficient
    names = ["const", "lg_d"] + list(extra) + list(dm.columns)
    X = np.column_stack([np.ones(len(P)), np.log10(P["d_km"].to_numpy())]
                        + [P[e].to_numpy(float) for e in extra] + [dm.to_numpy()])
    b, sg = tobit2(y, left, right, X, lo, hi)
    groups = [v.to_numpy() for v in P.reset_index(drop=True).groupby("tx").groups.values()]
    bb = []
    for _ in range(boot):
        idx = np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
        bb.append(tobit2(y[idx], left[idx], right[idx], X[idx], lo, hi)[0])
    se = np.std(bb, axis=0)
    coefs = pd.DataFrame({"term": names, "coef": b, "se_boot": se, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se})
    return dict(b=b, slope=b[1], se=se[1], sigma=sg, X=X, y=y, left=left, coefs=coefs)


def h4(D, out, L_all):
    res = {}
    M = c.link_table(D["ml"], D["nodes"])
    M = M[M["dist_ok"] & M["rssi"].notna()].copy()
    M["lg_d"] = np.log10(M["d_km"])
    M["zone"] = c.zone(M["mid_center_km"])
    M["rx_v4"] = M["rx_hw"].eq("HELTEC_V4").astype(int)
    res["n_links_rssi"] = len(M)
    rows = []
    if len(M) >= 20:
        f = smf.ols("rssi ~ lg_d + rx_v4", M).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(M["tx"])[0]})
        res.update(n_pooled=-f.params["lg_d"] / 10, n_pooled_se=f.bse["lg_d"] / 10, sigma_pooled=np.sqrt(f.scale),
                   a_pooled=f.params["Intercept"], b_pooled=f.params["lg_d"], rssi_v4_offset=f.params["rx_v4"],
                   n_links_v4=int(M["rx_v4"].sum()))
        rows.append(("RSSI, общая (+ сдвиг шлюзов V4)", len(M), res["n_pooled"], res["n_pooled_se"],
                     res["sigma_pooled"]))
        fe = fe_slope(M, "rssi", "lg_d")
        if fe:
            res.update(n_fe=-fe["b"] / 10, n_fe_se=fe["se"] / 10, sigma_fe=fe["sigma"], n_links_fe=fe["n_links"])
            rows.append(("RSSI, эффекты узлов", fe["n_links"], res["n_fe"], res["n_fe_se"], fe["sigma"]))
        for z, g in M.groupby("zone", observed=True):
            if len(g) >= 15:
                fz = smf.ols("rssi ~ lg_d + rx_v4" if g["rx_v4"].nunique() > 1 else "rssi ~ lg_d", g).fit()
                rows.append((f"RSSI, {z}", len(g), -fz.params["lg_d"] / 10, fz.bse["lg_d"] / 10, np.sqrt(fz.scale)))
    S = L_all.copy()
    S["zone"] = c.zone(S["mid_center_km"])
    f_ols = smf.ols("snr ~ lg_d", S).fit()
    rows.append(("SNR, МНК без поправок", len(S), -f_ols.params["lg_d"] / 10, f_ols.bse["lg_d"] / 10,
                 np.sqrt(f_ols.scale)))
    fes = fe_slope(S, "snr", "lg_d")
    if fes:
        res.update(n_snr_fe=-fes["b"] / 10, n_snr_fe_se=fes["se"] / 10, sigma_snr_fe=fes["sigma"],
                   n_links_snr_fe=fes["n_links"])
        rows.append(("SNR, эффекты узлов", fes["n_links"], res["n_snr_fe"], res["n_snr_fe_se"], fes["sigma"]))
    # Truncated regression: links inside the observable SNR window, hardware of both ends as covariates.
    lo, hi = SNR_TRUNC
    T = S[(S["snr"] > lo) & (S["snr"] < hi)].reset_index(drop=True)
    T_X = pd.get_dummies(T[["tx_model", "rx_model"]].fillna("OTHER"), drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(T)), T["lg_d"].to_numpy(), T_X.to_numpy()])
    if len(T) >= 50:
        b, sg = truncreg(T["snr"].to_numpy(), X, lo, hi)
        boots, txs = [], T["tx"].unique()
        groups = {k: v.to_numpy() for k, v in T.groupby("tx").groups.items()}
        for _ in range(100):  # cluster bootstrap over transmitters
            idx = np.concatenate([groups[k] for k in RNG.choice(txs, len(txs))])
            boots.append(truncreg(T["snr"].to_numpy()[idx], X[idx], lo, hi)[0][1])
        se = np.std(boots) / 10
        res.update(n_trunc=-b[1] / 10, n_trunc_se=se, sigma_trunc=sg, n_links_trunc=len(T))
        rows.append((f"SNR, усечённая регрессия ({fmt(lo, 0)}…{fmt(hi, 0)} дБ) + модели узлов", len(T),
                     res["n_trunc"], se, sg))
    # Censored design (MQTT): every packet of a sender uploaded by a gateway either came directly
    # (SNR known) or via relays (direct copy not decoded -> SNR below the limit).
    lo_c, hi_c = SNR_TRUNC
    Pk = c.mqtt_packets(D["mq"], D["nodes"])
    Pk = Pk[Pk["dist_ok"]].reset_index(drop=True)
    res.update(n_pk=len(Pk), n_pk_direct=int(Pk["direct"].sum()), n_pk_pairs=Pk.groupby(["tx", "rx"]).ngroups)
    Hk = Pk.dropna(subset=["tx_hag_m", "rx_hag_m"])
    res.update(n_pk_height=len(Hk), n_pk_height_direct=int(Hk["direct"].sum()))
    if len(Pk) >= 200 and Pk["direct"].sum() >= 30:
        cm = censored_pathloss(Pk, lo_c, hi_c)
        res.update(n_cens=-cm["slope"] / 10, n_cens_se=cm["se"] / 10, sigma_cens=cm["sigma"])
        cm["coefs"].to_csv(out / "g4_censored_model.csv", index=False)
        res["cens_coefs"] = cm["coefs"]
        # Г1 in the same censored design: heights of both ends.
        H = Hk.reset_index(drop=True)
        if len(H) >= 300 and H["direct"].sum() >= 50:
            for sd in ("tx", "rx"):
                H[f"lg_h_{sd}"] = np.log10(H[f"{sd}_hag_m"].clip(lower=2))
            ch = censored_pathloss(H, lo_c, hi_c, extra=("lg_h_tx", "lg_h_rx"))
            ch["coefs"].to_csv(out / "g1_censored_height.csv", index=False)
            q25, q75 = np.log10(np.percentile(H[["tx_hag_m", "rx_hag_m"]].clip(lower=2).values.ravel(), [25, 75]))
            cf = ch["coefs"].set_index("term")
            hw = [0.0] + [v for k, v in cf["coef"].items() if k.startswith("tx_model_")]
            res.update(cens_h_tx=cf.loc["lg_h_tx", "coef"], cens_h_tx_se=cf.loc["lg_h_tx", "se_boot"],
                       cens_h_rx=cf.loc["lg_h_rx", "coef"], cens_h_rx_se=cf.loc["lg_h_rx", "se_boot"],
                       cens_h_iqr=(cf.loc["lg_h_tx", "coef"] + cf.loc["lg_h_rx", "coef"]) * (q75 - q25),
                       cens_hag_q25=10 ** q25, cens_hag_q75=10 ** q75, cens_hw_range_tx=max(hw) - min(hw))
        rows.append((f"SNR, цензурированная (MQTT: прямой приём / через ретрансляторы), пакетов {len(Pk)}",
                     res["n_pk_pairs"], res["n_cens"], res["n_cens_se"], cm["sigma"]))
        from scipy.stats import norm
        Pk["p_model"] = norm.sf((lo_c - cm["X"] @ cm["b"]) / cm["sigma"])
        bins = np.array([0.3, 1, 2, 4, 8, 16, 32, 64, 100])
        bd = Pk.groupby(pd.cut(Pk["d_km"], bins), observed=True).agg(
            p_obs=("direct", "mean"), p_model=("p_model", "mean"), n=("direct", "size"))
        bd = bd[bd["n"] >= 10]
        bd.to_csv(out / "g4_direct_reception_by_distance.csv")
        res["direct_by_distance"] = bd
        fig, ax = plt.subplots(figsize=(6.8, 4))
        x = [iv.mid for iv in bd.index]
        k, n = bd["p_obs"] * bd["n"], bd["n"]
        zc = 1.96
        centre = (k + zc ** 2 / 2) / (n + zc ** 2)
        half = zc * np.sqrt(k * (n - k) / n + zc ** 2 / 4) / (n + zc ** 2)
        ax.errorbar(x, bd["p_obs"], yerr=[bd["p_obs"] - (centre - half), centre + half - bd["p_obs"]], fmt="o",
                    color=BLUE, ms=6, capsize=3, lw=1.5, label="доля пакетов, принятых шлюзом напрямую (95 % ДИ)")
        ax.plot(x, bd["p_model"], "-s", color=ORANGE, ms=5,
                label=f"цензурированная модель: n = {fmt(res['n_cens'], 2)}, σ = {fmt(cm['sigma'])} дБ")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("расстояние отправитель – шлюз, км")
        ax.set_ylabel("вероятность прямого приёма пакета")
        ax.set_title("Г4. Вероятность приёма по прямой линии от расстояния (MQTT)", fontsize=10, loc="left")
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(out / "fig_g4_link_pdr.png")
        plt.close(fig)
    for z, g in S.groupby("zone", observed=True):
        fz = fe_slope(g, "snr", "lg_d")
        if fz:
            rows.append((f"SNR, эффекты узлов, {z}", fz["n_links"], -fz["b"] / 10, fz["se"] / 10, fz["sigma"]))
    tab = pd.DataFrame(rows, columns=["оценка", "линий", "n", "SE(n)", "sigma, дБ"])
    tab.to_csv(out / "g4_pathloss.csv", index=False)
    res["table"] = tab

    # Range at the SF9/BW250 sensitivity limit, P = 20 dBm, 0 dBi.
    sens = c.THERMAL_250K + NF_DB + SNR_MIN_SF9
    res["sens_dbm"] = sens
    res["range_nir6_km"] = NIR6["d0_km"] * 10 ** ((P_TX_RU - NIR6["pl0"] - sens) / (10 * NIR6["n"]))
    if "b_pooled" in res:
        if res["n_pooled"] > 0.5:
            res["range_obs_km"] = 10 ** ((sens - res["a_pooled"]) / res["b_pooled"])
        fig, ax = plt.subplots(figsize=(6.8, 4))
        for sel, col, mk, lab in ((M["rx_v4"] == 0, BLUE, "o", "шлюз: прочие модели"),
                                  (M["rx_v4"] == 1, ORANGE, "s", "шлюз: Heltec V4")):
            ax.scatter(M.loc[sel, "d_km"], M.loc[sel, "rssi"], s=14, marker=mk, color=col, alpha=0.55, lw=0,
                       label=f"{lab} ({int(sel.sum())} линий)")
        dd = np.logspace(np.log10(M["d_km"].min()), np.log10(max(M["d_km"].max(), 30)), 50)
        ax.plot(dd, res["a_pooled"] + res["b_pooled"] * np.log10(dd), color=INK, lw=1.5,
                label=f"подгонка (прочие шлюзы): n = {fmt(res['n_pooled'], 2)}, σ = {fmt(res['sigma_pooled'])} дБ")
        ax.plot(dd, P_TX_RU - NIR6["pl0"] - 10 * NIR6["n"] * np.log10(dd / NIR6["d0_km"]), color=AQUA, ls="--",
                label="модель НИР6 (n = 2,08), P = 20 дБм")
        ax.axhline(sens, color=MUTED, lw=1, ls=":")
        ax.text(dd[0], sens + 1, f"чувствительность SF9/250 кГц ≈ {fmt(sens)} дБм", color=MUTED, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("расстояние, км")
        ax.set_ylabel("RSSI на шлюзе, дБм")
        ax.set_title("Г4. Затухание в Москве: RSSI прямых линий (MQTT)", fontsize=10, loc="left")
        ax.legend(loc="lower left", fontsize=7)
        fig.savefig(out / "fig_g4_pathloss.png")
        plt.close(fig)
    # SNR vs distance, binned, by zone (all link sources).
    fig, ax = plt.subplots(figsize=(6.8, 4))
    bins = np.logspace(np.log10(0.3), np.log10(S["d_km"].max() + 0.1), 12)
    for (z, g), col, mk in zip(S.groupby("zone", observed=True), (ORANGE, BLUE, AQUA), ("o", "s", "^")):
        med = g.groupby(pd.cut(g["d_km"], bins), observed=True)["snr"].agg(["median", "size"])
        med = med[med["size"] >= 5]
        ax.plot([iv.mid for iv in med.index], med["median"], "-" + mk, color=col, ms=6, label=f"{z}")
    ax.axhline(SNR_MIN_SF9, color=MUTED, lw=1, ls=":")
    ax.text(0.99, SNR_MIN_SF9 + 0.3, "предел демодуляции SF9", color=MUTED, fontsize=8, ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("расстояние, км")
    ax.set_ylabel("медиана SNR, дБ")
    ax.set_title("Г4. SNR прямых линий по зонам (все источники)", fontsize=10, loc="left")
    ax.legend(loc="upper right")
    fig.savefig(out / "fig_g4_snr_zones.png")
    plt.close(fig)
    return res


# ---------------------------------------------------------------- Г5

def h5(D, out, n_fit):
    """Receiver noise at MQTT gateways from packets with SNR <= NOISE_SNR_MAX.

    Above that the SX126x SNR estimate saturates (~+10…13 dB) and RSSI - SNR overestimates N.
    Heltec V4 gateways report RSSI with a large constant offset (front-end LNA), so the model of
    the gateway enters every comparison; `noise_adj` removes the fitted V4 offset.
    """
    mq, nodes = D["mq"], D["nodes"]
    m = mq[mq["radio_ok"] & (mq["rx_rssi"] < 0) & (mq["rx_snr"] <= NOISE_SNR_MAX)].copy()
    m["noise"] = c.noise_floor(m["rx_rssi"], m["rx_snr"])
    g = m.groupby("gw").agg(noise=("noise", "median"), n=("noise", "size"))
    g = g[g["n"] >= 10].join(nodes[["hw", "model", "hw_class", "dist_center_km", "hag_m"]])
    g["v4"] = g["hw"].eq("HELTEC_V4")
    g["zone"] = c.zone(g["dist_center_km"])
    thermal = c.THERMAL_250K + NF_DB
    res = {"n_gateways": len(g), "n_v4": int(g["v4"].sum()), "thermal_nf": thermal, "snr_max": NOISE_SNR_MAX}
    bym = g.groupby("model")["noise"].agg(["median", "size"])
    bym.to_csv(out / "g5_noise_by_gateway_model.csv")
    res["by_model"] = bym
    v4, oth = g.loc[g["v4"], "noise"], g.loc[~g["v4"] & g["hw"].notna() & g["hw"].ne("UNKNOWN"), "noise"]
    offset = 0.0
    if len(v4) >= 3 and len(oth) >= 3:
        offset = v4.median() - oth.median()
        bs = [np.median(RNG.choice(v4, len(v4))) - np.median(RNG.choice(oth, len(oth))) for _ in range(2000)]
        res.update(v4_offset=offset, v4_offset_ci=tuple(np.percentile(bs, [2.5, 97.5])),
                   v4_offset_p=stats.mannwhitneyu(v4, oth).pvalue, n_v4_gw=len(v4), n_oth_gw=len(oth))
    g["noise_adj"] = g["noise"] - offset * g["v4"]
    base = g.loc[~g["v4"], "noise"]
    if len(base):
        res.update(n_base=len(base), noise_median_base=base.median(), excess_median_base=base.median() - thermal,
                   share_above_3db_base=(base > thermal + 3).mean())
    gz = g.dropna(subset=["dist_center_km"])
    res["n_gateways_coords"] = len(gz)
    if len(gz) >= 8:
        f = smf.ols("noise ~ dist_center_km + v4", gz).fit(cov_type="HC3")
        lo, hi = f.conf_int().loc["dist_center_km"]
        res.update(noise_per_10km=10 * f.params["dist_center_km"], noise_per_10km_ci=(10 * lo, 10 * hi),
                   noise_dist_p=f.pvalues["dist_center_km"])
        res["rho_noise_center"], res["rho_p"] = stats.spearmanr(gz["noise_adj"], gz["dist_center_km"])
        zs = gz.groupby("zone", observed=True)["noise_adj"].agg(["median", "size"])
        res["zones"] = zs
        grp = [x["noise_adj"].to_numpy() for _, x in gz.groupby("zone", observed=True) if len(x) >= 3]
        if len(grp) >= 2:
            res["kw_zones_p"] = stats.kruskal(*grp).pvalue
        zs = zs[zs["size"] >= 3]
        if len(zs) >= 2 and n_fit and n_fit > 0.5:
            dn = zs["median"].iloc[0] - zs["median"].iloc[-1]
            res.update(dn_center_periphery=dn, dn_zones=(zs.index[0], zs.index[-1]),
                       range_factor=10 ** (-dn / (10 * n_fit)))
    # Does receiver noise explain link SNR beyond distance? (MQTT links with geometry)
    M = c.link_table(D["ml"], nodes)
    M = M[M["dist_ok"]].copy()
    M["lg_d"] = np.log10(M["d_km"])
    M["noise_rx"] = g["noise_adj"].reindex(M["rx"]).to_numpy()
    M = M.dropna(subset=["noise_rx"])
    if len(M) >= 30:
        f0 = smf.ols("snr ~ lg_d", M).fit()
        f1 = ols_2way("snr ~ lg_d + noise_rx", M)
        res.update(snr_noise_coef=f1.params["noise_rx"], snr_noise_ci=tuple(f1.conf_int().loc["noise_rx"]),
                   r2_dist=f0.rsquared, r2_dist_noise=f1.rsquared, n_links_noise=len(M))
    gout = g.reset_index(names="gateway").assign(gateway=lambda d: d["gateway"].map(c.anon))
    gout.to_csv(out / "g5_gateway_noise.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4))
    for sel, col, mk, lab in ((~gz["v4"], BLUE, "o", "шлюз: прочие модели"), (gz["v4"], ORANGE, "s", "шлюз: Heltec V4")):
        ax.scatter(gz.loc[sel, "dist_center_km"], gz.loc[sel, "noise"], s=28, marker=mk, color=col, alpha=0.75,
                   lw=0, label=f"{lab} (n = {int(sel.sum())})")
    ax.axhline(c.THERMAL_250K, color=MUTED, ls=":", lw=1)
    ax.axhline(thermal, color=AQUA, ls="--", lw=1.5, label=f"kTB + NF {fmt(NF_DB, 0)} дБ = {fmt(thermal)} дБм")
    ax.text(gz["dist_center_km"].max() if len(gz) else 1, c.THERMAL_250K - 1.5, "kTB (250 кГц) = −120 дБм",
            color=MUTED, fontsize=8, ha="right", va="top")
    ax.set_xlabel("расстояние шлюза от центра (Кремль), км")
    ax.set_ylabel("медиана N = RSSI − 10·lg(1+10^(SNR/10)), дБм")
    ax.set_title(f"Г5. Шум на приёмниках MQTT-шлюзов (пакеты с SNR ≤ {fmt(NOISE_SNR_MAX, 0)} дБ)", fontsize=10,
                 loc="left")
    ax.legend(loc="center right", fontsize=8)
    fig.savefig(out / "fig_g5_noise.png")
    plt.close(fig)
    return res, g["noise_adj"]


# ---------------------------------------------------------------- Г6

def h6(D, out):
    nodes, routes = D["nodes"], D["routes"]
    res = {}
    relay = pd.Series([r for x in routes for r in x["relays"] if r != c.BROADCAST], dtype="int64").value_counts()
    part = set()
    for x in routes:
        part.update([x["src"], x["dst"]] + [r for r in x["relays"] if r != c.BROADCAST])
    counts = relay.reindex(sorted(part), fill_value=0)
    res.update(n_routes=len(routes), n_participants=len(counts), n_relay_events=int(counts.sum()),
               n_relaying_nodes=int((counts > 0).sum()), gini_tr=gini(counts), top10_tr=top_share(counts, 0.1),
               top1_tr=top_share(counts, 0.01))
    # Null model: same number of relay events spread uniformly over the participants.
    null = [gini(RNG.multinomial(int(counts.sum()), np.ones(len(counts)) / len(counts))) for _ in range(200)]
    res["gini_tr_null"] = float(np.mean(null))
    # MQTT: last relayer of multi-hop packets.
    ml = D["ml"]
    mr = ml.loc[ml["how"] == "relay", "tx"].value_counts()
    if len(mr) >= 10:
        res.update(n_mqtt_relay_events=int(mr.sum()), n_mqtt_relays=len(mr), gini_mqtt=gini(mr),
                   top10_mqtt=top_share(mr, 0.1))

    # Backbone = top 10 % of participants by traceroute relay count.
    k = max(1, int(round(0.1 * len(counts))))
    top = counts.sort_values(ascending=False).index[:k]
    N = nodes.reindex(counts.index)
    N["relays"] = counts
    N["backbone"] = N.index.isin(top)
    bb, rest = N[N["backbone"]], N[~N["backbone"]]
    res["backbone_n"] = len(bb)
    hb, hr = bb["hag_m"].dropna(), rest["hag_m"].dropna()
    res.update(hag_bb=hb.median() if len(hb) else np.nan, hag_rest=hr.median() if len(hr) else np.nan,
               n_hag_bb=len(hb), n_hag_rest=len(hr))
    if len(hb) >= 5 and len(hr) >= 5:
        res["hag_mw_p"] = stats.mannwhitneyu(hb, hr, alternative="greater").pvalue
    routerish = N["role_name"].isin(["ROUTER", "ROUTER_LATE", "CLIENT_BASE", "REPEATER"])
    res["share_routerrole_bb"], res["share_routerrole_rest"] = routerish[N["backbone"]].mean(), routerish[~N["backbone"]].mean()
    tab = pd.crosstab(N["backbone"], routerish)
    if tab.shape == (2, 2):
        res["role_fisher_p"] = stats.fisher_exact(tab.values)[1]
    res["share_v4_bb"], res["share_v4_rest"] = (bb["hw"] == "HELTEC_V4").mean(), (rest["hw"] == "HELTEC_V4").mean()
    res["share_pocket_bb"], res["share_pocket_rest"] = (bb["hw_class"] == "pocket").mean(), (rest["hw_class"] == "pocket").mean()
    out_tab = N[["relays", "backbone", "model", "role_name", "hag_m", "dist_center_km"]].sort_values("relays", ascending=False)
    out_tab.reset_index(names="node").assign(node=lambda d: d["node"].map(c.anon)).to_csv(out / "g6_relay_load.csv", index=False)

    # Percolation on the undirected direct-link graph.
    G = nx.Graph()
    G.add_edges_from(D["obs"][["tx", "rx"]].drop_duplicates().itertuples(index=False))
    G.remove_nodes_from([c.BROADCAST])
    n0 = G.number_of_nodes()
    lcc0 = len(max(nx.connected_components(G), key=len))
    fr = np.linspace(0, 0.2, 21)

    def curve(order):
        y = []
        for f in fr:
            H = G.copy()
            H.remove_nodes_from(order[:int(round(f * n0))])
            y.append(len(max(nx.connected_components(H), key=len)) / n0 if H.number_of_nodes() else 0)
        return np.array(y)

    rel_order = sorted(G.nodes, key=lambda n: (-counts.get(n, 0), -G.degree(n)))
    deg_order = sorted(G.nodes, key=lambda n: -G.degree(n))
    y_rel, y_deg = curve(rel_order), curve(deg_order)
    rand = np.array([curve(list(RNG.permutation(list(G.nodes)))) for _ in range(20)])
    y_rand = rand.mean(axis=0)
    res.update(graph_nodes=n0, graph_edges=G.number_of_edges(), lcc0=lcc0 / n0)
    i5 = int(np.argmin(np.abs(fr - 0.05)))
    i10 = int(np.argmin(np.abs(fr - 0.10)))
    res.update(lcc_rel_5=y_rel[i5], lcc_rand_5=y_rand[i5], lcc_rel_10=y_rel[i10], lcc_rand_10=y_rand[i10],
               lcc_deg_10=y_deg[i10])
    pd.DataFrame({"removed_share": fr, "lcc_backbone_first": y_rel, "lcc_degree_first": y_deg,
                  "lcc_random_mean": y_rand}).to_csv(out / "g6_percolation.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axs[0]
    x, y = lorenz(counts)
    ax.plot(x, y, color=BLUE, label=f"трассировки: Джини {fmt(res['gini_tr'], 2)}")
    if "gini_mqtt" in res:
        x2, y2 = lorenz(mr)
        ax.plot(x2, y2, color=ORANGE, ls="--", label=f"MQTT, последний ретранслятор: {fmt(res['gini_mqtt'], 2)}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    ax.set_xlabel("доля узлов (по возрастанию нагрузки)")
    ax.set_ylabel("доля ретрансляций")
    ax.set_title("а) кривая Лоренца нагрузки ретрансляции", fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax = axs[1]
    ax.plot(fr * 100, y_rel, "-o", color=BLUE, ms=4, label="«хребет» первым")
    ax.plot(fr * 100, y_deg, "-s", color=AQUA, ms=4, label="по степени")
    ax.fill_between(fr * 100, rand.min(axis=0), rand.max(axis=0), color=ORANGE, alpha=0.15, lw=0)
    ax.plot(fr * 100, y_rand, "-^", color=ORANGE, ms=4, label="случайно (20 повторов)")
    ax.set_xlabel("удалено узлов, %")
    ax.set_ylabel("крупнейшая компонента, доля узлов")
    ax.set_title("б) перколяция графа прямых линий", fontsize=10, loc="left")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out / "fig_g6_backbone.png")
    plt.close(fig)
    return res


# ---------------------------------------------------------------- report

def p_txt(p):
    return "p < 0,001" if p is not None and p < 0.001 else f"p = {fmt(p, 3)}"


def write_summary(out, ov, r1, r2, r4, r5, r6):
    L = []
    w = L.append
    w(f"# НИР7 — первичный расчёт гипотез Г1–Г6\n")
    w(f"Сформировано {datetime.now(timezone.utc):%d.%m.%Y %H:%M} UTC скриптом `analysis/primary.py`. "
      "Расчёт **предварительный**: повторяется по мере сбора данных, числа и ДИ будут уточняться.\n")
    w("## Данные\n")
    w("| что | значение |\n|---|---|")
    w(f"| полный снимок ONEmesh | `{ov['snapshot']}` (история карты с 08.10.2026) |")
    w(f"| узлов в кэше / с координатами / с высотой над поверхностью | {ov['nodes']} / {ov['nodes_coords']} / {ov['nodes_hag']} |")
    w(f"| трассировок (уникальных) / наблюдений хопов с SNR | {ov['traceroutes_unique']} / {ov['traceroute_obs']} |")
    w(f"| наблюдений NeighborInfo | {ov['neighbour_obs']} |")
    w(f"| MQTT: записей, интервал | {ov['mqtt_records']}, {ov['mqtt_from']}–{ov['mqtt_to']} UTC ({fmt(ov['mqtt_hours'])} ч) |")
    w(f"| MQTT: шлюзов / наблюдений последнего хопа (из них zero-hop) | {ov['gateways']} / {ov['mqtt_link_obs']} ({ov['mqtt_zero_hop']}) |")
    w(f"| направленных прямых линий (все источники) | {ov['directed_links']} |")
    w(f"| проверка `relay_node`: у zero-hop пакетов равен младшему байту отправителя | {fmt(100 * ov['relay_byte_matches_sender'], 0)} % |")
    w(f"| многохоповых пакетов с однозначно определённым ретранслятором | {fmt(100 * ov['relay_resolved_share'], 0)} % |")
    w("")
    w("Линия — «узел A услышан узлом B напрямую»: хопы трассировок (SNR в обе стороны), NeighborInfo, "
      "последний хоп MQTT. Повторные наблюдения линии сводятся к медиане. Расстояние используется, только "
      "если оно ≥ 3× суммарной ячейки огрубления координат обоих узлов, ≥ 300 м и ≤ 100 км, а оба узла "
      f"неподвижны по истории координат (подвижных узлов: {ov['nodes_mobile']}). Высота над поверхностью = "
      "высота узла (MSL) − рельеф SRTM GL1 30 м (модель поверхности, частично включает застройку), допустимо −30…350 м.\n")

    w("## Г1. Высота установки влияет сильнее модели устройства\n")
    w("Оценки по спискам линий смещены к нулю отбором (см. Г4); основная — цензурированная модель MQTT ниже.\n")
    if "eff_height_iqr" in r1:
        w(f"Линий с известной высотой обоих концов: **{r1['n_links_height']}** ({r1['n_nodes_height']} узлов). "
          f"Модель: SNR ~ lg d + lg h_tx + lg h_rx + модель_tx + модель_rx, ошибки кластеризованы по обоим узлам.\n")
        w(f"- lg h передатчика: {fmt(r1['h_b_tx'])} ± {fmt(r1['h_se_tx'])} дБ/декаду ({p_txt(r1['h_p_tx'])}); "
          f"приёмника: {fmt(r1['h_b_rx'])} ± {fmt(r1['h_se_rx'])} ({p_txt(r1['h_p_rx'])}).")
        w(f"- Переход от 25-го к 75-му процентилю высоты ({fmt(r1['hag_q25'], 0)} → {fmt(r1['hag_q75'], 0)} м) на обоих "
          f"концах: **{fmt(r1['eff_height_iqr'])} дБ**; размах поправок моделей передатчика: **{fmt(r1['eff_hw_range_tx'])} дБ**.")
        w(f"- Частный R²: высота {fmt(r1['partial_r2_height'], 3)}, модели устройств {fmt(r1['partial_r2_hw'], 3)} "
          f"(R² полной модели {fmt(r1['r2_full'], 2)}).")
    else:
        w(f"Линий с высотой обоих концов мало ({r1.get('n_links_height', 0)}) — оценка отложена.")
    if "h1_b_tx" in r1:
        w(f"- Только высота передатчика (больше выборка, {r1['n_links_height_tx']} линий): "
          f"{fmt(r1['h1_b_tx'])} ± {fmt(r1['h1_se_tx'])} дБ/декаду ({p_txt(r1['h1_p_tx'])}).")
    if "cens_h_tx" in r4:
        w(f"- **Цензурированная модель MQTT** ({r4['n_pk_height']} пакетов, прямых {r4['n_pk_height_direct']}): "
          f"lg h_tx {fmt(r4['cens_h_tx'])} ± {fmt(r4['cens_h_tx_se'])}, lg h_rx {fmt(r4['cens_h_rx'])} ± "
          f"{fmt(r4['cens_h_rx_se'])} дБ/декаду; высота {fmt(r4['cens_hag_q25'], 0)} → {fmt(r4['cens_hag_q75'], 0)} м "
          f"на обоих концах: **{fmt(r4['cens_h_iqr'])} дБ** против размаха моделей передатчика "
          f"**{fmt(r4['cens_hw_range_tx'])} дБ**.")
    else:
        w(f"- Цензурированная модель MQTT с высотами: пока мало пакетов ({r4.get('n_pk_height', 0)}, прямых "
          f"{r4.get('n_pk_height_direct', 0)}; нужно ≥ 300 и ≥ 50).")
    w("\nРисунки: `fig_g1_height.png`, `fig_g3_hardware.png`; коэффициенты — `g1_g3_regressions.csv`, "
      "`g1_censored_height.csv`.\n")

    w("## Г2. Асимметрия линии выдаёт мощность передатчика\n")
    if "r2_cv" in r2:
        w(f"Двусторонних линий: {r2['n_pairs']}, в связной компоненте {r2['n_pairs_gc']} пар на {r2['n_nodes_gc']} узлах. "
          f"Медиана |SNR(A→B) − SNR(B→A)| = {fmt(r2['delta_abs_median'])} дБ, SD = {fmt(r2['delta_sd'])} дБ.\n")
        w(f"- Модель Δ_AB = q_A − q_B (q = P_tx + N, усиления антенн и потери трассы сокращаются) объясняет "
          f"R² = {fmt(r2['r2_in'], 2)} асимметрии на обучении и **R² = {fmt(r2['r2_cv'], 2)} вне выборки** "
          f"(5-кратная кросс-валидация, {r2['n_cv']} пар, r = {fmt(r2['r_cv'], 2)}; ридж λ = {fmt(r2['damp'])}, "
          f"без регуляризации R² = {fmt(r2['r2_cv_nodamp'], 2)}).")
        if "r2_cv_unsat" in r2:
            w(f"- Без пар с SNR > +10 дБ (насыщение оценки SNR): R² вне выборки = {fmt(r2['r2_cv_unsat'], 2)}.")
        if "v4_minus_v3" in r2:
            lo, hi = r2["v4_minus_v3_ci"]
            w(f"- Heltec V4 − V3 по медиане q: **{fmt(r2['v4_minus_v3'])} дБ** {fmt_ci(lo, hi)} "
              f"(n = {r2['n_v4']} / {r2['n_v3']}, Манн–Уитни {p_txt(r2['v4_v3_mw_p'])}).")
        if "kw_models_p" in r2:
            w(f"- Различие q между моделями (Краскел–Уоллис): {p_txt(r2['kw_models_p'])}.")
        if "pocket_minus_board_q" in r2:
            w(f"- Карманные − платы по q: {fmt(r2['pocket_minus_board_q'])} дБ ({p_txt(r2['pocket_board_mw_p'])}, "
              f"n карманных = {r2['n_pocket_q']}) — антенна в q не входит, ожидается ≈ 0.")
        if "q_noise_spearman" in r2:
            w(f"- Связь q с шумом на шлюзе (Г5): ρ = {fmt(r2['q_noise_spearman'], 2)} ({p_txt(r2['q_noise_p'])}, "
              f"n = {r2['n_q_noise']}).")
    else:
        w(f"Двусторонних линий мало ({r2.get('n_pairs', 0)}).")
    w("\nРисунок: `fig_g2_asymmetry.png`; таблицы — `g2_node_potentials.csv`, `g2_potential_by_model.csv`.\n")

    w("## Г3. Карманные устройства проигрывают N дБ\n")
    w("Оценки по спискам линий (трассировки, NeighborInfo, последний хоп MQTT) смещены к нулю тем же отбором, "
      "что и в Г4: существующая линия уже прошла отбор по запасу. Основная оценка — цензурированная модель MQTT.\n")
    w(f"Линий с хорошей геометрией: {r1['n_links_dist']}; карманных узлов в них: {r1['n_pocket_nodes_in_links']}. "
      "Модель: SNR ~ lg d + класс_tx + класс_rx (база — платы с внешней антенной).\n")
    for s, name in (("tx", "передатчик"), ("rx", "приёмник")):
        if f"pocket_{s}" in r1:
            lo, hi = r1[f"pocket_{s}_ci"]
            w(f"- Карманное устройство как {name}: **{fmt(r1[f'pocket_{s}'])} дБ** {fmt_ci(lo, hi)} ({p_txt(r1[f'pocket_{s}_p'])}).")
    if "sym_pocket" in r1:
        lo, hi = r1["sym_pocket_ci"]
        w(f"- Симметричная часть двусторонних линий (SNR_AB + SNR_BA)/2 на один карманный конец: "
          f"**{fmt(r1['sym_pocket'])} дБ** {fmt_ci(lo, hi)} ({p_txt(r1['sym_pocket_p'])}; "
          f"{r1['sym_n_pairs']} пар, с карманным концом {r1['sym_pairs_with_pocket']}).")
    elif "sym_n_pairs" in r1:
        w(f"- Симметричная часть двусторонних линий: из {r1['sym_n_pairs']} пар с карманным концом "
          f"{r1['sym_pairs_with_pocket']} — оценка отложена.")
    cc = r4.get("cens_coefs")
    if cc is not None:
        sel = cc[cc["term"].str.startswith(("tx_model_", "rx_model_"))]
        w(f"- Цензурированная модель MQTT (Г4; все пакеты пар отправитель–шлюз, поправки к SNR относительно "
          f"{c.SHORT[BASE_MODEL]}, 95 % ДИ бутстрепом): " + "; ".join(
              f"{c.SHORT.get(t.split('_model_')[1], t)} как {'передатчик' if t.startswith('tx') else 'приёмник'} "
              f"{fmt(r.coef)} {fmt_ci(r.ci_lo, r.ci_hi)}" for t, r in zip(sel['term'], sel.itertuples())) + ".")
    w("\nРисунок: `fig_g3_hardware.png` (поправки по моделям относительно Heltec V3); `g4_censored_model.csv`.\n")

    w("## Г4. Город затухает быстрее, чем модель НИР6\n")
    tab = r4.get("table")
    if tab is not None and len(tab):
        w("| оценка | линий | n | SE(n) | σ, дБ |\n|---|---|---|---|---|")
        for row in tab.itertuples(index=False):
            w(f"| {row[0]} | {row[1]} | {fmt(row[2], 2)} | {fmt(row[3], 2)} | {fmt(row[4])} |")
        w(f"\nНИР6 (Bor et al. 2016): n = 2,08, σ = 3,57 дБ, PL(40 м) = 127,41 дБ.")
    if "rssi_v4_offset" in r4:
        w(f"- Сдвиг RSSI на шлюзах Heltec V4 в той же регрессии: {fmt(r4['rssi_v4_offset'])} дБ "
          f"({r4['n_links_v4']} из {r4['n_links_rssi']} линий) — см. Г5.")
    if "range_obs_km" in r4:
        w(f"- Дальность до чувствительности SF9/250 кГц ({fmt(r4['sens_dbm'])} дБм): по наблюдаемой медианной линии "
          f"RSSI — {fmt(r4['range_obs_km'])} км; по модели НИР6 при P = 20 дБм и 0 дБи — {fmt(r4['range_nir6_km'], 2)} км.")
    else:
        w(f"- Модель НИР6 при P = 20 дБм и 0 дБи даёт дальность {fmt(r4['range_nir6_km'], 2)} км до чувствительности "
          f"SF9/250 кГц ({fmt(r4['sens_dbm'])} дБм); наблюдаемая линия RSSI пока слишком неопределённа для сравнения.")
    w("- **Списки линий (трассировки, NeighborInfo) для оценки затухания непригодны**: в них попадают только "
      "декодированные линии, длинная линия видна лишь при хорошем запасе, и наблюдаемый SNR почти не зависит от "
      "расстояния (строки «МНК», «эффекты узлов», «усечённая регрессия»). Нужны и неуслышанные передачи.")
    if "n_cens" in r4:
        w(f"- **Основная оценка — цензурированная (Тобит) по MQTT**: {r4['n_pk']} пакетов в {r4['n_pk_pairs']} парах "
          f"отправитель–шлюз, из них {r4['n_pk_direct']} приняты напрямую; пакет, пришедший на шлюз через "
          f"ретрансляторы, означает, что прямая копия не декодирована (SNR ниже {fmt(SNR_TRUNC[0], 0)} дБ). "
          f"n = **{fmt(r4['n_cens'], 2)} ± {fmt(r4['n_cens_se'], 2)}**, σ = **{fmt(r4['sigma_cens'])} дБ** "
          f"(SE — бутстреп по отправителям; σ включает и замирания отдельных пакетов). Вероятность прямого приёма "
          f"по расстоянию — `fig_g4_link_pdr.png`, `g4_direct_reception_by_distance.csv`.")
    else:
        w(f"- Цензурированная оценка по MQTT: пока мало пакетов с хорошей геометрией ({r4.get('n_pk', 0)}, "
          f"прямых {r4.get('n_pk_direct', 0)}); появится по мере сбора.")
    w("\nРисунки: `fig_g4_pathloss.png`, `fig_g4_snr_zones.png`; таблица — `g4_pathloss.csv`.\n")

    w("## Г5. В центре дальность ограничивает шум\n")
    w(f"Шум N = RSSI − 10·lg(1 + 10^(SNR/10)) по пакетам с SNR ≤ {fmt(r5['snr_max'], 0)} дБ (выше оценка SNR "
      f"насыщается и N завышается). Шлюзов с ≥ 10 такими пакетами: {r5['n_gateways']} (Heltec V4: {r5['n_v4']}).\n")
    if "v4_offset" in r5:
        lo, hi = r5["v4_offset_ci"]
        w(f"- **Шлюзы Heltec V4 показывают шум на {fmt(r5['v4_offset'])} дБ выше** прочих {fmt_ci(lo, hi)} "
          f"({p_txt(r5['v4_offset_p'])}; {r5['n_v4_gw']} против {r5['n_oth_gw']} шлюзов) — систематический сдвиг "
          f"показаний RSSI (усилитель входного модуля V4 не учтён в RSSI), а не реальный шум. RSSI от V4 нельзя "
          f"смешивать с другими моделями; дальше сдвиг вычитается.")
    if "noise_median_base" in r5:
        w(f"- Шлюзы прочих моделей ({r5['n_base']}): медиана {fmt(r5['noise_median_base'])} дБм против "
          f"kTB + NF = {fmt(r5['thermal_nf'])} дБм: превышение **{fmt(r5['excess_median_base'])} дБ**; "
          f"выше на 3+ дБ — {fmt(100 * r5['share_above_3db_base'], 0)} % шлюзов.")
    if "noise_per_10km" in r5:
        lo, hi = r5["noise_per_10km_ci"]
        w(f"- Шум vs расстояние от центра (с поправкой на V4, {r5['n_gateways_coords']} шлюзов): "
          f"{fmt(r5['noise_per_10km'])} дБ на 10 км {fmt_ci(lo, hi)} ({p_txt(r5['noise_dist_p'])}); "
          f"ρ Спирмена = {fmt(r5['rho_noise_center'], 2)} ({p_txt(r5['rho_p'])}).")
        zs = r5["zones"]
        w("- По зонам (медиана с поправкой, дБм): " + "; ".join(f"{z}: {fmt(v['median'])} (n = {int(v['size'])})"
                                                             for z, v in zs.iterrows()) +
          (f"; Краскел–Уоллис {p_txt(r5['kw_zones_p'])}." if "kw_zones_p" in r5 else "."))
        if "range_factor" in r5:
            z0, z1 = r5["dn_zones"]
            w(f"- Разница «{z0}» − «{z1}»: {fmt(r5['dn_center_periphery'])} дБ ⇒ при найденном n дальность "
              f"×{fmt(r5['range_factor'], 2)} только из-за шума.")
    if "snr_noise_coef" in r5:
        lo, hi = r5["snr_noise_ci"]
        w(f"- SNR линии ~ lg d + шум приёмника: коэффициент при шуме {fmt(r5['snr_noise_coef'], 2)} {fmt_ci(lo, hi, 2)} "
          f"(теория −1); R² {fmt(r5['r2_dist'], 2)} → {fmt(r5['r2_dist_noise'], 2)} ({r5['n_links_noise']} линий).")
    w("\nРисунок: `fig_g5_noise.png`; таблицы — `g5_gateway_noise.csv`, `g5_noise_by_gateway_model.csv`.\n")

    w("## Г6. Сеть держится на «хребте»\n")
    w(f"Трассировок с полным путём: {r6['n_routes']}; узлов на путях: {r6['n_participants']}, ретранслировали "
      f"{r6['n_relaying_nodes']}; событий ретрансляции {r6['n_relay_events']}.\n")
    w(f"- Джини нагрузки ретрансляции: **{fmt(r6['gini_tr'], 2)}** (при равномерной нагрузке ≈ {fmt(r6['gini_tr_null'], 2)}); "
      f"10 % узлов выполняют **{fmt(100 * r6['top10_tr'], 0)} %** ретрансляций, 1 % — {fmt(100 * r6['top1_tr'], 0)} %.")
    if "gini_mqtt" in r6:
        w(f"- По MQTT (последний ретранслятор, {r6['n_mqtt_relay_events']} пакетов, {r6['n_mqtt_relays']} узлов): "
          f"Джини {fmt(r6['gini_mqtt'], 2)}, топ-10 % — {fmt(100 * r6['top10_mqtt'], 0)} %.")
    w(f"- «Хребет» ({r6['backbone_n']} узлов): медианная высота над поверхностью {fmt(r6['hag_bb'], 0)} м "
      f"(n = {r6['n_hag_bb']}) против {fmt(r6['hag_rest'], 0)} м (n = {r6['n_hag_rest']})"
      + (f", {p_txt(r6['hag_mw_p'])} (односторонний МУ)" if "hag_mw_p" in r6 else "") +
      f"; роли ROUTER/ROUTER_LATE/CLIENT_BASE — {fmt(100 * r6['share_routerrole_bb'], 0)} % против "
      f"{fmt(100 * r6['share_routerrole_rest'], 0)} %" + (f" (Фишер {p_txt(r6['role_fisher_p'])})" if "role_fisher_p" in r6 else "") +
      f"; Heltec V4 — {fmt(100 * r6['share_v4_bb'], 0)} % против {fmt(100 * r6['share_v4_rest'], 0)} %.")
    w(f"- Перколяция (граф прямых линий: {r6['graph_nodes']} узлов, {r6['graph_edges']} рёбер, крупнейшая компонента "
      f"{fmt(100 * r6['lcc0'], 0)} %): после удаления 10 % узлов «хребта» остаётся {fmt(100 * r6['lcc_rel_10'], 0)} %, "
      f"по степени — {fmt(100 * r6['lcc_deg_10'], 0)} %, случайных — {fmt(100 * r6['lcc_rand_10'], 0)} %.")
    w("\nРисунок: `fig_g6_backbone.png`; таблицы — `g6_relay_load.csv`, `g6_percolation.csv`.\n")
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            continue
        if isinstance(v, tuple):
            v = [x.item() if isinstance(x, (np.floating, np.integer)) else x for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", help="full snapshot dir (default: newest with DONE)")
    ap.add_argument("--partial", action="store_true", help="allow a snapshot without DONE (debugging)")
    args = ap.parse_args()
    t0 = time.time()
    style()
    full = c.Path(args.snapshot) if args.snapshot else c.latest_full()
    done = (full / "DONE").exists()
    if not done and not args.partial:
        raise SystemExit(f"{full} has no DONE (use --partial for a debugging run)")
    D = build(full)
    out = c.RESULTS / (full.name + ("" if done else "_partial"))
    out.mkdir(parents=True, exist_ok=True)
    ov = overview(D)
    print("data:", json.dumps(_jsonable(ov), ensure_ascii=False), flush=True)
    r5, gw_noise = h5(D, out, None)
    r1, L = h1_h3(D, out)
    r2, _ = h2(D, out, gw_noise)
    r4 = h4(D, out, L)
    cc = r4.get("cens_coefs")
    if cc is not None:
        t = cc[cc["term"].str.startswith(("tx_model_", "rx_model_"))]
        hw = pd.DataFrame({"side": t["term"].str[:2], "model": t["term"].str.split("_model_").str[1],
                           "coef": t["coef"], "lo": t["ci_lo"], "hi": t["ci_hi"]})
        plot_hardware(hw, out, "цензурированная модель MQTT")
    else:
        plot_hardware(r1["hw_linklist"], out, "списки линий, смещено к 0 отбором")
    if r5.get("zones") is not None and "n_fe" in r4:  # range factor needs n: recompute with the fitted value
        r5, _ = h5(D, out, r4["n_fe"])
    r6 = h6(D, out)
    write_summary(out, ov, r1, r2, r4, r5, r6)
    allres = {"overview": ov, "g1_g3": r1, "g2": r2, "g4": r4, "g5": r5, "g6": r6}
    (out / "numbers.json").write_text(json.dumps({k: _jsonable(v) for k, v in allres.items()},
                                                 ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
