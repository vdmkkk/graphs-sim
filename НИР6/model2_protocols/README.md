# Model 2 — Protocol-suitability simulator for LoRa mesh networks

This model evaluates **routing protocols / network stacks** as candidate
foundations for decentralised offline communication over LoRa. Unlike Model 1
(which scores path-finding *algorithms* by computational effort), Model 2 runs
each protocol on a physically-grounded, event-driven simulation of a real
LoRa mesh and reports the metrics that decide field viability.

---

## 1. Why a second model?

A protocol is more than its routing algorithm. On LoRa, suitability is
dominated by constraints the algorithm never sees:

* **Airtime is scarce.** A frame at SF12 occupies the channel for ~1 s; the
  radio is half-duplex and the medium is shared, so every control message
  competes with payload.
* **Regulation caps transmission.** EU868 allows each node only a **1 %
  duty cycle** — ~36 s of airtime per hour. A chatty protocol simply cannot
  send what it wants to.
* **Links are lossy and range-limited.** Reachability is set by geography and
  the SNR a spreading factor needs, not by an abstract adjacency list.
* **Nodes are constrained and unreliable.** Battery drains, nodes sleep, fail,
  and rejoin.

Model 2 makes all of this explicit so the comparison reflects *deployability*,
not just theoretical path quality.

---

## 2. Physical model (every constant is justified)

All constants live in [`config.py`](config.py) with citations. Highlights:

### 2.1 LoRa PHY ([`radio/lora_phy.py`](radio/lora_phy.py))
* **Time-on-air** — exact Semtech SX1276 formula (datasheet §4.1.1.7) from
  SF, bandwidth, coding rate and payload length. This is the cost that
  duty-cycle and collisions both ration.
* **Receiver sensitivity & required SNR** per spreading factor (SF7…SF12),
  from the SX127x datasheet.
* **Noise floor** = −174 dBm/Hz + 10·log₁₀(BW) + noise figure.

### 2.2 Propagation ([`config.py` → `Propagation`])
Log-distance path loss with log-normal shadowing:

```
PL(d) = PL(d₀) + 10·n·log₁₀(d/d₀) + X,   X ~ N(0, σ²)
```

with `PL(d₀)=127.41 dB`, `d₀=40 m`, `n=2.08`, `σ=3.57 dB` — the values
**measured for LoRa at 868 MHz** by Bor, Roedig, Voigt & Alonso, *"Do LoRa
Low-Power Wide-Area Networks Scale?"* (MSWiM 2016). At SF9/14 dBm this yields a
reliable range of ≈330 m (printed at startup).

### 2.3 Channel ([`radio/channel.py`](radio/channel.py))
A shared half-duplex medium modelled at frame granularity:
* **Collisions** between time-overlapping frames at a common receiver.
* **Capture effect** — the stronger frame survives if it exceeds every
  interferer by ≥ 6 dB; otherwise both are lost (the LoRaSim/FLoRa model).
* **Probabilistic reception** — a logistic ramp around the SNR threshold, so
  marginal links fail intermittently rather than at a brick wall.

### 2.4 The "weight", made physical
Routing cost of a link is the classic **ETX scaled by airtime**:

```
cost(u→v) = time_on_air(v) / P_delivery(u→v)      ("expected airtime per delivery")
```

A link is cheap only when it is *both* reliable and fast — exactly what a real
LoRa router should minimise. Two link notions are kept on purpose
([`network/deployment.py`](network/deployment.py)):
* **audible** (≥ sensitivity) — what the channel uses for reception *and*
  interference, so flooding's opportunistic long shots and collisions are both
  modelled;
* **routing-quality** (mean delivery prob ≥ 0.7) — what routing tables and the
  oracle build paths over, because real link-quality routing does not use
  50/50 links.

### 2.5 Nodes ([`network/node.py`](network/node.py))
Stateful, unlike Model 1's inert vertices: battery (energy = I·V·t over
TX/RX/idle/sleep currents), a sliding-window **duty-cycle budget**, half-duplex
radio state, and scheduled **failure/recovery**.

### 2.6 MAC ([`sim/engine.py`](sim/engine.py))
One MAC for all protocols, so only routing differs:
* **CSMA / listen-before-talk** with randomised backoff,
* **EU868 1 % duty-cycle enforcement** (a frame that would breach the budget is
  dropped — this is what punishes chatty protocols),
* **link-layer ARQ** (ACK + retransmission) for *unicast* frames; broadcast
  flood frames are never ACKed because their reliability comes from redundancy.

---

## 3. Protocols compared ([`protocols/`](protocols/))

| Protocol | Class | Idea | Expected LoRa trade-off |
|---|---|---|---|
| **ManagedFlood** | flooding | Meshtastic-style: rebroadcast once, SNR-based contention window, overhear-suppression | robust, zero control, redundant airtime |
| **EpidemicGossip** | flooding | probabilistic rebroadcast (DTN) | most robust under churn, most wasteful |
| **AODV** | reactive | on-demand RREQ/RREP, unicast routes | low idle overhead, per-flow discovery latency, fragile under churn |
| **OLSR** | proactive | HELLO + MPR-flooded TC, precomputed routes | routes always ready, but control traffic collides with the duty cycle |
| **Ideal** | oracle | instant perfect topology, least-cost path, **no control, no discovery** | the physical ceiling — still bound by collisions/duty/half-duplex |

The **Ideal** baseline is the headline reference: the gap between each real
protocol and Ideal answers "how much of the physically-possible performance
does this protocol actually capture?".

---

## 4. Metrics ([`metrics/collector.py`](metrics/collector.py))

* **PDR** — message delivery ratio (headline reliability).
* **Latency** — end-to-end delay of delivered messages (mean / median / p95).
* **Goodput** — delivered application payload per second.
* **Overhead** — control-frame count, control airtime fraction, transmissions
  per delivery (redundancy), control frames per delivery.
* **Channel** — airtime utilisation, collision rate, duty-cycle / busy drops.
* **Energy** — total joules, **energy per delivered bit**, mean node lifetime,
  first-node-death time.
* **Resilience** — PDR retention under churn / partition vs. the baseline.

---

## 5. Scenarios ([`scenarios/`](scenarios/))

* **baseline** — static network, moderate load.
* **churn** — nodes fail and rejoin (flat batteries, movement, obstructions).
* **partition** — a large fraction fails at once, splitting the network.
* **load sweep** — offered load is increased to map the **latency-vs-throughput**
  trade-off and find the congestion-collapse point.

Deployment layouts: `UNIFORM`, `CLUSTERED` (villages/camps), `CORRIDOR`
(road/valley, near-linear), `GRID`.

---

## 6. Running it

```bash
# from the repository root, run as a package
python -m model2_protocols.main --quick            # ~10 s smoke run
python -m model2_protocols.main                    # full study
python -m model2_protocols.main --sf 12 --nodes 60 --time 3600 --reps 5
python -m model2_protocols.main --no-sweep         # skip the load sweep
```

Outputs land in `model2_protocols/results/`:

| File | Contents |
|---|---|
| `results_matrix.csv` | every (protocol × scenario × repetition) run |
| `results_load_sweep.csv` | per-offered-load runs |
| `summary_by_protocol_scenario.csv` | aggregated comparison table |
| `pdr_pivot.csv`, `latency_pivot.csv` | quick pivots |
| `resilience.csv` | PDR retention under churn / partition |
| `pdr_vs_load.png` | PDR vs offered load |
| `latency_vs_throughput.png` | the latency-throughput trade-off curve |
| `pdr_by_scenario.png` | resilience bar chart |

Runs are **deterministic** for a fixed `--seed` (verified): the physical
network, traffic and node dynamics depend only on the seed, so all protocols
are compared on the *same* network and workload.

---

## 7. Reading the results for the NIR

A representative full run supports this narrative:

* **Managed flooding sits closest to the physical ceiling** with *zero* control
  overhead and the lowest latency — which is exactly why Meshtastic ships it.
* **Epidemic gossip is the most resilient under churn**, bought with the most
  redundant airtime and energy.
* **AODV** pays a heavy discovery cost (control airtime ≫ payload) and degrades
  sharply under churn as routes break — classic MANET assumptions do not hold
  on a duty-cycle-limited radio.
* **OLSR's** proactive HELLO/TC traffic **collides with the 1 % duty cycle**
  (visible as duty-cycle drops and the shortest node lifetime): periodic
  link-state maintenance is structurally mismatched to LoRa.

Conclusion direction: for decentralised offline LoRa communication, **controlled
flooding is the most suitable foundation**, with the oracle quantifying the
headroom a smarter-but-still-cheap protocol could hope to recover.

> Numbers depend on configuration (SF, density, load, intervals); all are CLI-
> or `config.py`-tunable. Treat the *ordering and mechanisms* as the result,
> and report the exact config used for any quoted figure.
