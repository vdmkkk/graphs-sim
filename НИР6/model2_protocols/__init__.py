"""Model 2 — Protocol-suitability simulator for LoRa mesh networks.

Where Model 1 benchmarks *algorithms* as mathematical objects, Model 2
evaluates whether a *routing protocol / network stack* is a viable basis
for decentralised offline communication over LoRa.

It does so with:
  * a physically-grounded LoRa PHY (log-distance path loss, SNR, time-on-air),
  * a discrete-event MAC with a shared half-duplex channel, collisions and
    capture effect, listen-before-talk and EU868 duty-cycle limits,
  * stateful nodes (battery / energy model, duty-cycle budget, failures),
  * five routing strategies spanning the real design space, and
  * network-level metrics (PDR, end-to-end latency, goodput, control
    overhead, energy-per-delivered-bit, network lifetime, resilience).
"""
