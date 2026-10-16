# LoRa mesh simulation suite — two complementary models

Supporting code for the research work *"Экспериментальная оценка
децентрализованной передачи сообщений в LoRa mesh-сетях для
оффлайн-коммуникаций"* (Experimental evaluation of decentralised message
delivery in LoRa mesh networks for offline communication).

The study separates **two questions** that are easy to conflate, and answers
each with its own model. They live in clearly separated folders.

```
graphs-sim/
├── model1_algorithms/     # Model 1 — algorithms as mathematical objects
│   ├── algorithms/        #   Dijkstra, Bellman-Ford, A*, BFS, Bi-Dijkstra, Floyd-Warshall
│   ├── graphs/            #   abstract topology generators (line, ring, mesh, ...)
│   ├── runner.py  main.py
│   ├── analytics/  graph_images/  results.csv
│   └── ...
└── model2_protocols/      # Model 2 — protocols / network stacks for real LoRa mesh
    ├── radio/             #   LoRa PHY: path loss, SNR, time-on-air, channel + collisions
    ├── network/           #   geographic deployment, stateful nodes, frames
    ├── sim/               #   discrete-event engine + shared MAC (CSMA, duty cycle, ARQ)
    ├── protocols/         #   ManagedFlood, Epidemic, AODV, OLSR, Ideal baseline
    ├── metrics/  scenarios/
    ├── runner.py  main.py
    └── README.md          #   ← detailed design + physical justifications
```

## Model 1 — algorithmic efficiency (`model1_algorithms/`)

Every routing protocol ultimately rests on a path-finding **algorithm**.
Model 1 isolates that layer: it benchmarks classic shortest-path algorithms on
ten abstract graph topologies and measures *computational* effort — nodes
visited, edges relaxed, run time, path optimality. Edge weights are abstract
(uniform random), because the question here is purely "how good is the
algorithm as a mathematical object?".

```bash
cd model1_algorithms
python main.py                  # -> results.csv + analytics/ + graph_images/
```

## Model 2 — protocol suitability (`model2_protocols/`)

Model 2 climbs up a layer: it asks whether a **protocol / network stack** is a
viable basis for decentralised offline communication *on real LoRa hardware*.
It replaces abstract weights with a physically-grounded radio model and a
discrete-event MAC, gives nodes real state (battery, duty-cycle budget,
failures), and measures *communication* outcomes instead of CPU effort:
packet delivery ratio, end-to-end latency, goodput, control overhead, energy
per delivered bit, network lifetime and resilience to outages.

```bash
cd model2_protocols           # run as a package from the repo root:
python -m model2_protocols.main --quick          # fast smoke run (~10 s)
python -m model2_protocols.main                  # full study (matrix + load sweep)
python -m model2_protocols.main --sf 12 --nodes 60
```

See [model2_protocols/README.md](model2_protocols/README.md) for the full
design, the justification behind every physical constant, and how to read the
results for the paper.

## How the two models tell one story

| | Model 1 | Model 2 |
|---|---|---|
| **Unit of study** | algorithm | protocol / network stack |
| **Weights** | abstract (random) | physical (airtime ÷ delivery prob = ETX) |
| **Topology** | abstract graphs | geographic placement + LoRa range |
| **Nodes** | inert vertices | battery, duty cycle, failures |
| **Channel** | none | half-duplex, collisions, capture, duty cycle |
| **Metrics** | nodes visited, edges relaxed, time | PDR, latency, throughput, overhead, energy |
| **Answer** | which algorithm is computationally best | which protocol actually works on LoRa |

Model 1 establishes the theoretical core; Model 2 shows what survives contact
with the physics of a low-power, duty-cycle-limited, lossy radio mesh.

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`model2_protocols/results/` (generated tables and plots) is git-ignored
because it is fully reproducible from a single command; delete that line from
`.gitignore` if you want to track a particular run alongside the paper.
