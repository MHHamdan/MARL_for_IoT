"""
Perception Task Simulation for IoT Edge Computing Environment

Simulates visual perception tasks (object detection, classification)
with different model choices offering accuracy-latency trade-offs.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class PerceptionModel:
    """Specification for a perception model."""
    name: str
    accuracy: float  # Base accuracy (mAP or similar)
    latency_factor: float  # Relative latency (1.0 = baseline)
    energy_factor: float  # Relative energy (1.0 = baseline)
    flops_million: float  # Computational cost in MFLOPS
    input_size: Tuple[int, int] = (224, 224)
    output_size: int = 1000  # Number of classes or detection boxes


class PerceptionTaskGenerator:
    """
    Generates perception tasks with varying characteristics.

    Simulates:
    - Object detection tasks
    - Classification tasks
    - Frame importance (some frames more critical than others)
    """

    def __init__(
        self,
        models: List[Dict],
        base_data_size_kb: float = 150.0,
        base_compute_cycles: float = 1e9,
        deadline_base_ms: float = 100.0
    ):
        """
        Initialize perception task generator.

        Args:
            models: List of model specifications
            base_data_size_kb: Base data size for input frame in KB
            base_compute_cycles: Base compute cycles for processing
            deadline_base_ms: Base deadline in milliseconds
        """
        self.models = self._parse_models(models)
        self.base_data_size_kb = base_data_size_kb
        self.base_compute_cycles = base_compute_cycles
        self.deadline_base_ms = deadline_base_ms

        # Task counter
        self.task_counter = 0

        # Scene complexity factors
        self.complexity_mean = 1.0
        self.complexity_std = 0.3

    def _parse_models(self, model_configs: List[Dict]) -> List[PerceptionModel]:
        """Parse model configurations into PerceptionModel objects."""
        models = []
        for config in model_configs:
            models.append(PerceptionModel(
                name=config['name'],
                accuracy=config['accuracy'],
                latency_factor=config['latency_factor'],
                energy_factor=config['energy_factor'],
                flops_million=config.get('flops_million', 100)
            ))
        return models

    def get_model(self, model_idx: int) -> PerceptionModel:
        """Get model by index."""
        return self.models[model_idx]

    def get_num_models(self) -> int:
        """Get number of available models."""
        return len(self.models)

    def generate_task(
        self,
        device_id: int,
        model_idx: int,
        frame_rate: float = 1.0,
        scene_complexity: Optional[float] = None
    ) -> Dict:
        """
        Generate a perception task.

        Args:
            device_id: ID of the IoT device generating the task
            model_idx: Index of selected perception model
            frame_rate: Frame sampling rate (0-1, affects deadline)
            scene_complexity: Optional complexity factor (default: random)

        Returns:
            Task dictionary with all task parameters
        """
        self.task_counter += 1
        model = self.models[model_idx]

        # Scene complexity affects computation and accuracy
        if scene_complexity is None:
            scene_complexity = np.clip(
                np.random.normal(self.complexity_mean, self.complexity_std),
                0.5, 2.0
            )

        # Data size based on complexity
        data_size_bits = self.base_data_size_kb * 1024 * 8 * scene_complexity

        # Compute cycles based on model and complexity
        compute_cycles = (
            self.base_compute_cycles *
            model.latency_factor *
            scene_complexity
        )

        # Deadline based on frame rate
        # Higher frame rate = tighter deadline
        deadline_ms = self.deadline_base_ms / max(frame_rate, 0.1)

        # Task importance/priority
        priority = int(np.random.exponential(1.0))

        return {
            'task_id': self.task_counter,
            'device_id': device_id,
            'model_idx': model_idx,
            'data_size_bits': data_size_bits,
            'compute_cycles': compute_cycles,
            'deadline_ms': deadline_ms,
            'scene_complexity': scene_complexity,
            'priority': priority,
            'model_name': model.name,
            'base_accuracy': model.accuracy
        }

    def simulate_accuracy(
        self,
        model_idx: int,
        scene_complexity: float,
        resource_factor: float = 1.0,
        latency_factor: float = 1.0
    ) -> float:
        """
        Simulate perception accuracy based on model and conditions.

        Args:
            model_idx: Index of perception model
            scene_complexity: Scene complexity (higher = harder)
            resource_factor: Resource allocation (0-1)
            latency_factor: How close to deadline (1.0 = met, <1.0 = rushed)

        Returns:
            Simulated accuracy value
        """
        model = self.models[model_idx]

        # Base accuracy
        accuracy = model.accuracy

        # Complexity penalty (harder scenes reduce accuracy)
        complexity_penalty = 1.0 - 0.1 * (scene_complexity - 1.0)
        accuracy *= complexity_penalty

        # Resource factor (less resources = potential accuracy drop)
        if resource_factor < 0.5:
            resource_penalty = 0.5 + 0.5 * (resource_factor / 0.5)
            accuracy *= resource_penalty

        # Latency factor (rushing reduces accuracy)
        if latency_factor < 1.0:
            latency_penalty = 0.7 + 0.3 * latency_factor
            accuracy *= latency_penalty

        # Add small noise
        accuracy += np.random.normal(0, 0.02)

        return np.clip(accuracy, 0.0, 1.0)

    def get_model_specs(self) -> List[Dict]:
        """Get specifications for all models."""
        return [
            {
                'idx': i,
                'name': m.name,
                'accuracy': m.accuracy,
                'latency_factor': m.latency_factor,
                'energy_factor': m.energy_factor,
                'flops_million': m.flops_million
            }
            for i, m in enumerate(self.models)
        ]


class FrameSampler:
    """
    Simulates adaptive frame sampling for perception tasks.

    Implements policies for selecting which frames to process
    based on content importance and resource availability.
    """

    def __init__(
        self,
        max_frame_rate: float = 30.0,
        min_frame_rate: float = 1.0
    ):
        """
        Initialize frame sampler.

        Args:
            max_frame_rate: Maximum frames per second
            min_frame_rate: Minimum frames per second
        """
        self.max_frame_rate = max_frame_rate
        self.min_frame_rate = min_frame_rate

        # Frame buffer for temporal analysis
        self.frame_buffer: List[Dict] = []
        self.buffer_size = 10

    def compute_frame_importance(
        self,
        motion_score: float,
        scene_change_score: float,
        detection_count: int
    ) -> float:
        """
        Compute importance score for a frame.

        Args:
            motion_score: Amount of motion (0-1)
            scene_change_score: Scene change indicator (0-1)
            detection_count: Number of objects detected

        Returns:
            Importance score (0-1)
        """
        # Weighted combination
        importance = (
            0.3 * motion_score +
            0.3 * scene_change_score +
            0.4 * min(detection_count / 10, 1.0)
        )
        return np.clip(importance, 0.0, 1.0)

    def adaptive_frame_rate(
        self,
        target_rate: float,
        importance: float,
        resource_availability: float
    ) -> float:
        """
        Compute adaptive frame rate based on importance and resources.

        Args:
            target_rate: Target frame rate (fraction of max)
            importance: Frame importance score
            resource_availability: Available resources (0-1)

        Returns:
            Adjusted frame rate in fps
        """
        # Base rate from target
        base_rate = self.min_frame_rate + target_rate * (
            self.max_frame_rate - self.min_frame_rate
        )

        # Adjust based on importance
        importance_factor = 0.5 + 0.5 * importance

        # Adjust based on resources
        resource_factor = 0.3 + 0.7 * resource_availability

        adjusted_rate = base_rate * importance_factor * resource_factor

        return np.clip(adjusted_rate, self.min_frame_rate, self.max_frame_rate)

    def should_process_frame(
        self,
        current_rate: float,
        frame_importance: float,
        time_since_last: float
    ) -> bool:
        """
        Decide whether to process current frame.

        Args:
            current_rate: Current frame rate in fps
            frame_importance: Current frame importance
            time_since_last: Time since last processed frame in seconds

        Returns:
            True if frame should be processed
        """
        # Minimum interval based on rate
        min_interval = 1.0 / current_rate

        # Always process if exceeded minimum interval
        if time_since_last >= min_interval:
            return True

        # Process early if very important frame
        if frame_importance > 0.8 and time_since_last >= min_interval * 0.5:
            return True

        return False


class IoTDevice:
    """
    Simulates an IoT device with perception capabilities.

    Handles:
    - Frame generation
    - Local processing capability
    - Battery management
    - Task generation
    """

    def __init__(
        self,
        device_id: int,
        cpu_frequency_ghz: float = 1.5,
        battery_capacity_mah: float = 5000.0,
        transmit_power_dbm: float = 20.0
    ):
        """
        Initialize IoT device.

        Args:
            device_id: Unique device identifier
            cpu_frequency_ghz: Local CPU frequency
            battery_capacity_mah: Battery capacity
            transmit_power_dbm: Transmission power
        """
        self.device_id = device_id
        self.cpu_frequency_ghz = cpu_frequency_ghz
        self.battery_capacity_mah = battery_capacity_mah
        self.transmit_power_dbm = transmit_power_dbm

        # Current state
        self.battery_level = 1.0  # 0-1 fraction
        self.current_model_idx = 0
        self.current_frame_rate = 1.0  # 0-1 fraction of max

        # Task queue
        self.pending_tasks: List[Dict] = []
        self.max_queue_size = 10

        # Position (for channel model)
        self.position = np.random.uniform(0, 100, size=2)

        # Metrics
        self.total_tasks_generated = 0
        self.total_energy_consumed = 0.0

    def reset(self):
        """Reset device state."""
        self.battery_level = 1.0
        self.current_model_idx = 0
        self.current_frame_rate = 1.0
        self.pending_tasks.clear()
        self.total_tasks_generated = 0
        self.total_energy_consumed = 0.0

    def generate_task(
        self,
        task_generator: PerceptionTaskGenerator,
        model_idx: Optional[int] = None,
        frame_rate: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Generate a new perception task.

        Args:
            task_generator: Task generator instance
            model_idx: Optional model index (uses current if None)
            frame_rate: Optional frame rate (uses current if None)

        Returns:
            Generated task or None if queue is full
        """
        if len(self.pending_tasks) >= self.max_queue_size:
            return None

        model_idx = model_idx if model_idx is not None else self.current_model_idx
        frame_rate = frame_rate if frame_rate is not None else self.current_frame_rate

        task = task_generator.generate_task(
            device_id=self.device_id,
            model_idx=model_idx,
            frame_rate=frame_rate
        )

        self.pending_tasks.append(task)
        self.total_tasks_generated += 1

        return task

    def local_process(
        self,
        task: Dict,
        task_generator: PerceptionTaskGenerator
    ) -> Dict:
        """
        Process task locally on device.

        Args:
            task: Task to process
            task_generator: For accuracy simulation

        Returns:
            Processing results
        """
        # Local compute time
        compute_cycles = task['compute_cycles']
        compute_time_s = compute_cycles / (self.cpu_frequency_ghz * 1e9)
        compute_time_ms = compute_time_s * 1000

        # Local energy (CPU power ~1W for mobile device)
        energy_j = compute_time_s * 1.0

        # Simulate accuracy
        accuracy = task_generator.simulate_accuracy(
            model_idx=task['model_idx'],
            scene_complexity=task['scene_complexity'],
            resource_factor=1.0,  # Full local resources
            latency_factor=min(1.0, task['deadline_ms'] / compute_time_ms)
        )

        # Update battery
        self.battery_level -= energy_j / (self.battery_capacity_mah * 3.6)
        self.battery_level = max(0.0, self.battery_level)
        self.total_energy_consumed += energy_j

        return {
            'latency_ms': compute_time_ms,
            'energy_j': energy_j,
            'accuracy': accuracy,
            'deadline_violated': compute_time_ms > task['deadline_ms'],
            'processed_locally': True
        }

    def get_observation(self) -> np.ndarray:
        """
        Get observation vector for this device.

        Returns:
            Observation array containing device state
        """
        obs = np.array([
            self.battery_level,
            len(self.pending_tasks) / self.max_queue_size,
            self.current_model_idx / 3.0,  # Normalized by max models
            self.current_frame_rate,
            self.cpu_frequency_ghz / 3.0  # Normalized
        ], dtype=np.float32)
        # Sanitize observation
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    def get_info(self) -> Dict:
        """Get device information dictionary."""
        return {
            'device_id': self.device_id,
            'battery_level': self.battery_level,
            'queue_length': len(self.pending_tasks),
            'current_model': self.current_model_idx,
            'frame_rate': self.current_frame_rate,
            'total_tasks': self.total_tasks_generated,
            'total_energy': self.total_energy_consumed
        }
