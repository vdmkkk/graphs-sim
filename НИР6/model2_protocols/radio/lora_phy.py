"""LoRa physical layer: time-on-air, path loss, RSSI/SNR and reception model.

These functions turn a *geographic* link (distance in metres) into the
quantities that actually drive routing decisions and metrics:

    distance --PL--> RSSI --(noise)--> SNR --(SF limit)--> link quality

This is what makes Model 2's edge weights physically justified instead of
random: a "good link" is one with a comfortable SNR margin and low airtime,
not an arbitrary small integer.
"""

from __future__ import annotations

import math

from ..config import LoRaPHY, Propagation

# Thermal noise floor: kTB at room temperature => -174 dBm/Hz.
THERMAL_NOISE_DBM_PER_HZ = -174.0


def time_on_air(payload_bytes: int, phy: LoRaPHY) -> float:
    """Semtech time-on-air (seconds) for a LoRa frame.

    Implements the SX1276 datasheet formula (section 4.1.1.7). This is the
    single most important cost in a LoRa network: airtime is the scarce
    resource that duty-cycle regulation and collisions both ration.
    """
    sf = phy.sf
    bw = phy.bandwidth_hz
    cr = phy.coding_rate
    de = 1 if (phy.low_dr_optimize if phy.low_dr_optimize is not None else sf >= 11) else 0
    ih = 0 if phy.explicit_header else 1
    crc = 1 if phy.crc else 0

    t_sym = (2 ** sf) / bw
    t_preamble = (phy.preamble_symbols + 4.25) * t_sym

    numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * crc - 20 * ih
    denominator = 4 * (sf - 2 * de)
    payload_sym = 8 + max(math.ceil(numerator / denominator) * (cr + 4), 0)

    t_payload = payload_sym * t_sym
    return t_preamble + t_payload


def noise_floor_dbm(phy: LoRaPHY) -> float:
    """Receiver noise floor for the configured bandwidth and noise figure."""
    return THERMAL_NOISE_DBM_PER_HZ + 10 * math.log10(phy.bandwidth_hz) + phy.noise_figure_db


def path_loss_db(distance_m: float, prop: Propagation, shadowing_db: float = 0.0) -> float:
    """Log-distance path loss (dB), optional sampled shadowing term."""
    d = max(distance_m, 1.0)
    return prop.pl_d0_db + 10 * prop.path_loss_exp * math.log10(d / prop.d0_m) + shadowing_db


def rssi_dbm(distance_m: float, phy: LoRaPHY, prop: Propagation, shadowing_db: float = 0.0) -> float:
    """Received signal strength (dBm) = EIRP - path loss + rx antenna gain."""
    eirp = phy.tx_power_dbm + phy.antenna_gain_db
    return eirp - path_loss_db(distance_m, prop, shadowing_db) + phy.antenna_gain_db


def snr_db(distance_m: float, phy: LoRaPHY, prop: Propagation, shadowing_db: float = 0.0) -> float:
    """Signal-to-noise ratio (dB) at the receiver for a given distance."""
    return rssi_dbm(distance_m, phy, prop, shadowing_db) - noise_floor_dbm(phy)


def reception_probability(snr: float, rssi: float, phy: LoRaPHY) -> float:
    """Probability a frame is decoded, given SNR and RSSI.

    Two hard gates (sensitivity floor, SNR limit) plus a soft logistic ramp
    around the SNR threshold that models residual symbol errors near the
    edge of coverage rather than a brick-wall cliff.
    """
    if rssi < phy.sensitivity():
        return 0.0
    margin = snr - phy.required_snr()
    if margin <= -2.0:
        return 0.0
    if margin >= 6.0:
        return 1.0
    # Logistic ramp: ~0.5 at +1 dB margin, steepness ~1.2/dB.
    return 1.0 / (1.0 + math.exp(-1.2 * (margin - 1.0)))


class LoRaLink:
    """A directed radio link between two fixed points.

    Caches the deterministic part (mean SNR) and exposes airtime so the
    routing layer can build *physically meaningful* link costs.
    """

    __slots__ = ("distance_m", "phy", "prop", "mean_rssi", "mean_snr")

    def __init__(self, distance_m: float, phy: LoRaPHY, prop: Propagation):
        self.distance_m = distance_m
        self.phy = phy
        self.prop = prop
        self.mean_rssi = rssi_dbm(distance_m, phy, prop)
        self.mean_snr = snr_db(distance_m, phy, prop)

    def is_feasible(self) -> bool:
        """Is the mean link even theoretically usable (above sensitivity)?"""
        return self.mean_rssi >= self.phy.sensitivity() and self.mean_snr >= self.phy.required_snr()

    def mean_reception_probability(self) -> float:
        return reception_probability(self.mean_snr, self.mean_rssi, self.phy)

    def airtime(self, payload_bytes: int) -> float:
        return time_on_air(payload_bytes, self.phy)

    def snr_margin(self) -> float:
        return self.mean_snr - self.phy.required_snr()
