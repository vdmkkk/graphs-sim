"""Central configuration: physical constants and tunable simulation parameters.

Every magic number that has a real-world meaning lives here with a short
justification and a citation hint, so the weights used by Model 2 are
*explainable* rather than arbitrary (the key requirement that separates it
from Model 1, where edge weights were uniform random integers).

References
----------
* Semtech SX1276 datasheet — time-on-air, receiver sensitivity, SNR limits.
* Bor, Roedig, Voigt, Alonso (2016) "Do LoRa Low-Power Wide-Area Networks
  Scale?" — log-distance path-loss parameters measured for LoRa.
* ETSI EN 300 220 — EU868 sub-band 1% duty-cycle obligation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# LoRa PHY — per spreading-factor characteristics (BW = 125 kHz, CR = 4/5)
# ---------------------------------------------------------------------------

# Required SNR at the demodulator for each spreading factor (Semtech SX127x).
# Lower SF = faster but needs a cleaner link; higher SF = slower but reaches
# further. This trade-off is the heart of LoRa link planning.
SNR_REQUIRED_DB: dict[int, float] = {
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0,
}

# Receiver sensitivity (dBm) for BW=125 kHz, derived from the same datasheet.
SENSITIVITY_DBM: dict[int, float] = {
    7: -123.0,
    8: -126.0,
    9: -129.0,
    10: -132.0,
    11: -134.5,
    12: -137.0,
}


@dataclass(frozen=True)
class LoRaPHY:
    """Physical-layer radio configuration shared by every node."""

    sf: int = 9                    # spreading factor (7..12)
    bandwidth_hz: float = 125e3    # 125 kHz — the common EU868 channel
    coding_rate: int = 1           # CR 4/(4+CR); 1 => 4/5
    preamble_symbols: int = 8      # standard LoRaWAN preamble
    explicit_header: bool = True
    crc: bool = True
    low_dr_optimize: bool | None = None  # None => auto-enable for SF>=11

    tx_power_dbm: float = 14.0     # EU868 legal max for most sub-bands
    antenna_gain_db: float = 2.0   # modest omni gain at each end
    noise_figure_db: float = 6.0   # receiver noise figure

    def required_snr(self) -> float:
        return SNR_REQUIRED_DB[self.sf]

    def sensitivity(self) -> float:
        return SENSITIVITY_DBM[self.sf]


# ---------------------------------------------------------------------------
# Propagation — log-distance path-loss model (Bor et al. 2016, LoRa @ 868 MHz)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Propagation:
    """Log-distance path loss with log-normal shadowing.

    PL(d) = PL_d0 + 10 * n * log10(d / d0) + X,  X ~ N(0, sigma^2)
    """

    pl_d0_db: float = 127.41   # path loss at reference distance d0
    d0_m: float = 40.0         # reference distance (metres)
    path_loss_exp: float = 2.08   # environment exponent n (measured, suburban)
    shadowing_sigma_db: float = 3.57  # log-normal shadowing std-dev


# ---------------------------------------------------------------------------
# Energy model — current draw per radio state (typical SX1276 + MCU board)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyModel:
    """Battery and per-state current draw. Energy = I * V * t."""

    voltage_v: float = 3.3
    battery_mah: float = 2000.0     # a single 18650-class cell

    current_tx_ma: float = 120.0    # PA active at +14 dBm
    current_rx_ma: float = 11.0     # receiver listening
    current_idle_ma: float = 1.6    # MCU awake, radio standby
    current_sleep_ma: float = 0.05  # deep sleep between duties

    def joules(self, current_ma: float, seconds: float) -> float:
        return self.voltage_v * (current_ma / 1000.0) * seconds


# ---------------------------------------------------------------------------
# MAC / regulatory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MACConfig:
    """Channel-access and duty-cycle parameters."""

    duty_cycle: float = 0.01          # EU868 sub-band g1 — 1 %
    duty_window_s: float = 3600.0     # sliding window for duty accounting
    listen_before_talk: bool = True   # CSMA — sense before transmitting
    cad_seconds: float = 0.01         # channel-activity-detection dwell
    backoff_min_s: float = 0.02       # random backoff lower bound
    backoff_max_s: float = 0.30       # random backoff upper bound
    max_backoff_tries: int = 8        # give up after this many busy senses
    capture_threshold_db: float = 6.0  # co-channel capture margin
    proc_delay_s: float = 0.02        # per-hop store-and-forward processing

    # Link-layer ARQ for *unicast* frames (DATA / RREP). Broadcast/flood
    # frames are never ACKed — flooding's redundancy is its reliability.
    use_arq: bool = True
    ack_bytes: int = 5                # tiny acknowledgement frame
    sifs_s: float = 0.05              # short gap before the receiver ACKs
    ack_timeout_s: float = 1.0        # wait for ACK before retransmitting
    max_data_retx: int = 3            # retransmissions before giving up


# ---------------------------------------------------------------------------
# Bundled scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Everything one simulation run needs, in one place."""

    phy: LoRaPHY = field(default_factory=LoRaPHY)
    propagation: Propagation = field(default_factory=Propagation)
    energy: EnergyModel = field(default_factory=EnergyModel)
    mac: MACConfig = field(default_factory=MACConfig)

    seed: int = 42
    sim_time_s: float = 3600.0        # one hour of network operation

    # Header sizes (bytes) used to size control vs payload airtime/energy.
    data_payload_bytes: int = 32      # a short text message
    header_bytes: int = 16            # routing + addressing overhead

    @property
    def sim_time(self) -> float:
        """Alias used throughout the runner / metrics for sim duration."""
        return self.sim_time_s

    def clone(self, **overrides) -> "SimConfig":
        """Shallow copy with field overrides (handy for parameter sweeps)."""
        import copy

        new = copy.deepcopy(self)
        for key, value in overrides.items():
            setattr(new, key, value)
        return new
