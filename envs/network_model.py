"""
Wireless Channel Model for IoT Edge Computing Environment

Implements Rayleigh fading channel with path loss model for realistic
wireless communication simulation between IoT devices and edge servers.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ChannelConfig:
    """Configuration for wireless channel model."""
    bandwidth_mhz: float = 20.0
    noise_power_dbm: float = -100.0
    path_loss_exponent: float = 3.5
    reference_distance_m: float = 1.0
    reference_path_loss_db: float = 30.0
    rayleigh_scale: float = 1.0


class WirelessChannel:
    """
    Wireless channel model implementing Rayleigh fading with path loss.

    The channel model computes:
    1. Path loss based on distance
    2. Rayleigh fading for small-scale fading
    3. Signal-to-Noise Ratio (SNR)
    4. Achievable data rate using Shannon capacity
    """

    def __init__(self, config: Optional[ChannelConfig] = None):
        """
        Initialize wireless channel model.

        Args:
            config: Channel configuration parameters
        """
        self.config = config or ChannelConfig()
        self.bandwidth_hz = self.config.bandwidth_mhz * 1e6
        self.noise_power_w = self._dbm_to_watt(self.config.noise_power_dbm)

        # Cache for current channel states
        self._channel_gains = None
        self._snr_values = None

    def _dbm_to_watt(self, power_dbm: float) -> float:
        """Convert power from dBm to Watts."""
        return 10 ** ((power_dbm - 30) / 10)

    def _watt_to_dbm(self, power_w: float) -> float:
        """Convert power from Watts to dBm."""
        return 10 * np.log10(power_w) + 30

    def compute_path_loss(self, distance_m: float) -> float:
        """
        Compute path loss in dB using log-distance model.

        PL(d) = PL(d0) + 10 * n * log10(d/d0)

        Args:
            distance_m: Distance in meters

        Returns:
            Path loss in dB
        """
        if distance_m <= 0:
            distance_m = 0.1  # Minimum distance

        path_loss_db = (
            self.config.reference_path_loss_db +
            10 * self.config.path_loss_exponent *
            np.log10(distance_m / self.config.reference_distance_m)
        )
        return path_loss_db

    def generate_rayleigh_fading(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate Rayleigh fading coefficients.

        The magnitude of the fading coefficient follows Rayleigh distribution.
        |h|^2 follows exponential distribution with mean = 2 * scale^2

        Args:
            num_samples: Number of fading samples to generate

        Returns:
            Array of fading power gains (|h|^2)
        """
        # Generate complex Gaussian samples
        real_part = np.random.normal(0, self.config.rayleigh_scale, num_samples)
        imag_part = np.random.normal(0, self.config.rayleigh_scale, num_samples)

        # Rayleigh fading power gain
        fading_gain = real_part**2 + imag_part**2
        return fading_gain

    def compute_snr(
        self,
        transmit_power_dbm: float,
        distance_m: float,
        fading_gain: Optional[float] = None
    ) -> float:
        """
        Compute Signal-to-Noise Ratio.

        SNR = (P_tx * PL^(-1) * |h|^2) / N0

        Args:
            transmit_power_dbm: Transmit power in dBm
            distance_m: Distance in meters
            fading_gain: Optional fading gain (if None, generates new sample)

        Returns:
            SNR in linear scale
        """
        # Transmit power in Watts
        tx_power_w = self._dbm_to_watt(transmit_power_dbm)

        # Path loss in linear scale
        path_loss_db = self.compute_path_loss(distance_m)
        path_loss_linear = 10 ** (path_loss_db / 10)

        # Fading gain
        if fading_gain is None:
            fading_gain = self.generate_rayleigh_fading(1)[0]

        # Received power
        rx_power_w = tx_power_w * fading_gain / path_loss_linear

        # SNR
        snr = rx_power_w / self.noise_power_w
        return snr

    def compute_snr_db(
        self,
        transmit_power_dbm: float,
        distance_m: float,
        fading_gain: Optional[float] = None
    ) -> float:
        """Compute SNR in dB."""
        snr_linear = self.compute_snr(transmit_power_dbm, distance_m, fading_gain)
        return 10 * np.log10(max(snr_linear, 1e-10))

    def compute_data_rate(
        self,
        transmit_power_dbm: float,
        distance_m: float,
        fading_gain: Optional[float] = None,
        bandwidth_fraction: float = 1.0
    ) -> float:
        """
        Compute achievable data rate using Shannon capacity.

        R = B * log2(1 + SNR)

        Args:
            transmit_power_dbm: Transmit power in dBm
            distance_m: Distance in meters
            fading_gain: Optional fading gain
            bandwidth_fraction: Fraction of bandwidth allocated (0-1)

        Returns:
            Data rate in bits per second
        """
        snr = self.compute_snr(transmit_power_dbm, distance_m, fading_gain)
        effective_bandwidth = self.bandwidth_hz * bandwidth_fraction
        data_rate = effective_bandwidth * np.log2(1 + snr)
        return data_rate

    def compute_transmission_time(
        self,
        data_size_bits: float,
        transmit_power_dbm: float,
        distance_m: float,
        fading_gain: Optional[float] = None,
        bandwidth_fraction: float = 1.0
    ) -> float:
        """
        Compute time to transmit data over the channel.

        Args:
            data_size_bits: Size of data to transmit in bits
            transmit_power_dbm: Transmit power in dBm
            distance_m: Distance in meters
            fading_gain: Optional fading gain
            bandwidth_fraction: Fraction of bandwidth allocated

        Returns:
            Transmission time in seconds
        """
        data_rate = self.compute_data_rate(
            transmit_power_dbm, distance_m, fading_gain, bandwidth_fraction
        )
        if data_rate <= 0:
            return float('inf')
        return data_size_bits / data_rate

    def update_channel_state(
        self,
        num_devices: int,
        num_servers: int,
        device_positions: np.ndarray,
        server_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update channel state for all device-server pairs.

        Args:
            num_devices: Number of IoT devices
            num_servers: Number of edge servers
            device_positions: Device positions shape (num_devices, 2)
            server_positions: Server positions shape (num_servers, 2)

        Returns:
            Tuple of (channel_gains, snr_matrix) both shape (num_devices, num_servers)
        """
        # Generate fading gains for all links
        self._channel_gains = self.generate_rayleigh_fading(num_devices * num_servers)
        self._channel_gains = self._channel_gains.reshape(num_devices, num_servers)

        # Compute SNR for all links
        self._snr_values = np.zeros((num_devices, num_servers))

        for i in range(num_devices):
            for j in range(num_servers):
                distance = np.linalg.norm(device_positions[i] - server_positions[j])
                self._snr_values[i, j] = self.compute_snr(
                    transmit_power_dbm=20,  # Default device transmit power
                    distance_m=max(distance, 1.0),
                    fading_gain=self._channel_gains[i, j]
                )

        return self._channel_gains, self._snr_values

    def get_channel_quality(self, device_idx: int, server_idx: int) -> float:
        """
        Get normalized channel quality for a device-server pair.

        Returns value in [0, 1] where 1 is excellent channel quality.
        """
        if self._snr_values is None:
            return 0.5  # Default medium quality

        snr_db = 10 * np.log10(max(self._snr_values[device_idx, server_idx], 1e-10))

        # Normalize SNR to [0, 1] range
        # Typical SNR range: -10 dB (poor) to 30 dB (excellent)
        normalized = (snr_db + 10) / 40
        return np.clip(normalized, 0.0, 1.0)


class ChannelSimulator:
    """
    Simulates channel conditions over time with temporal correlation.

    Uses Jakes' model for temporal correlation in Rayleigh fading.
    """

    def __init__(
        self,
        config: Optional[ChannelConfig] = None,
        doppler_freq_hz: float = 10.0,
        sample_interval_s: float = 0.01
    ):
        """
        Initialize channel simulator.

        Args:
            config: Channel configuration
            doppler_freq_hz: Maximum Doppler frequency (related to mobility)
            sample_interval_s: Time between channel samples
        """
        self.channel = WirelessChannel(config)
        self.doppler_freq = doppler_freq_hz
        self.sample_interval = sample_interval_s

        # Temporal correlation coefficient
        self.rho = self._compute_correlation()

        # Previous channel gains for temporal correlation
        self._prev_gains = None

    def _compute_correlation(self) -> float:
        """Compute temporal correlation using Jakes' model."""
        # J0(2 * pi * fd * Ts) where J0 is Bessel function of first kind
        from scipy.special import j0
        return j0(2 * np.pi * self.doppler_freq * self.sample_interval)

    def step(self, num_links: int) -> np.ndarray:
        """
        Generate next channel gains with temporal correlation.

        Args:
            num_links: Number of communication links

        Returns:
            Array of channel gains for each link
        """
        # Generate innovation (new random component)
        innovation = self.channel.generate_rayleigh_fading(num_links)

        if self._prev_gains is None:
            self._prev_gains = innovation
            return innovation

        # Apply temporal correlation
        # h[n] = rho * h[n-1] + sqrt(1-rho^2) * w[n]
        new_gains = (
            self.rho * self._prev_gains +
            np.sqrt(1 - self.rho**2) * innovation
        )

        self._prev_gains = new_gains
        return np.abs(new_gains)  # Ensure positive gains

    def reset(self):
        """Reset channel state."""
        self._prev_gains = None
