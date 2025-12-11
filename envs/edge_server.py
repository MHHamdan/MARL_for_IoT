"""
Edge Server Simulation for IoT Edge Computing Environment

Implements edge server with computation resources, task queue management,
and resource allocation capabilities.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque


@dataclass
class ServerConfig:
    """Configuration for edge server."""
    cpu_frequency_ghz: float = 3.0
    num_cores: int = 8
    memory_gb: float = 16.0
    max_concurrent_tasks: int = 10
    power_idle_w: float = 50.0
    power_per_core_w: float = 20.0


@dataclass
class Task:
    """Represents a computation task."""
    task_id: int
    device_id: int
    data_size_bits: float
    compute_cycles: float
    deadline_ms: float
    perception_model_idx: int
    arrival_time: float
    priority: int = 0

    # Results (filled after processing)
    completion_time: Optional[float] = None
    accuracy: Optional[float] = None
    energy_consumed: Optional[float] = None
    was_dropped: bool = False


class EdgeServer:
    """
    Edge server simulation with computation resources and task management.

    Handles:
    - Task queue management
    - Resource allocation
    - Task execution simulation
    - Performance metrics tracking
    """

    def __init__(self, server_id: int, config: Optional[ServerConfig] = None):
        """
        Initialize edge server.

        Args:
            server_id: Unique identifier for this server
            config: Server configuration parameters
        """
        self.server_id = server_id
        self.config = config or ServerConfig()

        # Task queue
        self.task_queue: deque = deque(maxlen=100)
        self.active_tasks: List[Task] = []

        # Resource tracking
        self.cpu_utilization = 0.0  # 0-1 fraction
        self.memory_utilization = 0.0  # 0-1 fraction
        self.allocated_cores = 0

        # Performance metrics
        self.total_tasks_processed = 0
        self.total_tasks_dropped = 0
        self.total_energy_consumed = 0.0
        self.total_latency = 0.0

        # Current simulation time
        self.current_time = 0.0

        # Connected devices
        self.connected_devices: List[int] = []

    def reset(self):
        """Reset server state."""
        self.task_queue.clear()
        self.active_tasks.clear()
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.allocated_cores = 0
        self.total_tasks_processed = 0
        self.total_tasks_dropped = 0
        self.total_energy_consumed = 0.0
        self.total_latency = 0.0
        self.current_time = 0.0

    def add_task(self, task: Task) -> bool:
        """
        Add a task to the server queue.

        Args:
            task: Task to add

        Returns:
            True if task was added, False if queue is full
        """
        if len(self.task_queue) >= self.task_queue.maxlen:
            task.was_dropped = True
            self.total_tasks_dropped += 1
            return False

        task.arrival_time = self.current_time
        self.task_queue.append(task)
        return True

    def get_queue_length(self) -> int:
        """Get current queue length."""
        return len(self.task_queue)

    def get_available_resources(self) -> Dict[str, float]:
        """Get available resource fractions."""
        return {
            'cpu': 1.0 - self.cpu_utilization,
            'memory': 1.0 - self.memory_utilization,
            'cores': self.config.num_cores - self.allocated_cores
        }

    def allocate_resources(self, cpu_fraction: float, memory_fraction: float) -> bool:
        """
        Allocate resources for task processing.

        Args:
            cpu_fraction: Fraction of CPU to allocate (0-1)
            memory_fraction: Fraction of memory to allocate (0-1)

        Returns:
            True if allocation successful
        """
        new_cpu = self.cpu_utilization + cpu_fraction
        new_memory = self.memory_utilization + memory_fraction

        if new_cpu > 1.0 or new_memory > 1.0:
            return False

        self.cpu_utilization = new_cpu
        self.memory_utilization = new_memory
        return True

    def release_resources(self, cpu_fraction: float, memory_fraction: float):
        """Release allocated resources."""
        self.cpu_utilization = max(0.0, self.cpu_utilization - cpu_fraction)
        self.memory_utilization = max(0.0, self.memory_utilization - memory_fraction)

    def compute_execution_time(
        self,
        task: Task,
        allocated_cpu_fraction: float
    ) -> float:
        """
        Compute task execution time based on allocated resources.

        Execution time = cycles / (frequency * cores * efficiency)

        Args:
            task: Task to execute
            allocated_cpu_fraction: Fraction of CPU allocated

        Returns:
            Execution time in milliseconds
        """
        # Effective computation capacity
        effective_cores = allocated_cpu_fraction * self.config.num_cores
        effective_frequency = self.config.cpu_frequency_ghz * 1e9  # Hz

        # Cycles per second
        compute_capacity = effective_frequency * max(effective_cores, 0.1)

        # Execution time in seconds, then convert to ms
        execution_time_s = task.compute_cycles / compute_capacity
        return execution_time_s * 1000

    def compute_energy_consumption(
        self,
        execution_time_ms: float,
        allocated_cpu_fraction: float
    ) -> float:
        """
        Compute energy consumption for task execution.

        Energy = Power * Time
        Power = idle_power + active_cores * power_per_core

        Args:
            execution_time_ms: Execution time in milliseconds
            allocated_cpu_fraction: Fraction of CPU allocated

        Returns:
            Energy consumption in Joules
        """
        active_cores = allocated_cpu_fraction * self.config.num_cores
        power = self.config.power_idle_w + active_cores * self.config.power_per_core_w
        execution_time_s = execution_time_ms / 1000
        return power * execution_time_s

    def process_task(
        self,
        task: Task,
        allocated_cpu_fraction: float,
        model_accuracy: float
    ) -> Dict[str, float]:
        """
        Process a task and return results.

        Args:
            task: Task to process
            allocated_cpu_fraction: Fraction of CPU to use
            model_accuracy: Base accuracy of the perception model

        Returns:
            Dictionary with processing results
        """
        # Compute execution time
        execution_time = self.compute_execution_time(task, allocated_cpu_fraction)

        # Add queue wait time
        queue_wait_time = len(self.active_tasks) * 5  # 5ms per queued task (simplified)

        # Total latency
        total_latency = execution_time + queue_wait_time

        # Check deadline
        deadline_violated = total_latency > task.deadline_ms

        # Compute energy
        energy = self.compute_energy_consumption(execution_time, allocated_cpu_fraction)

        # Simulate accuracy (affected by resource allocation)
        # Lower resources = potential accuracy degradation
        accuracy_factor = min(1.0, 0.5 + 0.5 * allocated_cpu_fraction)
        final_accuracy = model_accuracy * accuracy_factor

        # Update task
        task.completion_time = self.current_time + total_latency / 1000
        task.accuracy = final_accuracy
        task.energy_consumed = energy

        # Update server stats
        self.total_tasks_processed += 1
        self.total_latency += total_latency
        self.total_energy_consumed += energy

        return {
            'latency_ms': total_latency,
            'energy_j': energy,
            'accuracy': final_accuracy,
            'deadline_violated': deadline_violated,
            'execution_time_ms': execution_time,
            'queue_wait_ms': queue_wait_time
        }

    def step(self, time_delta_s: float) -> List[Dict[str, Any]]:
        """
        Advance simulation by time delta and process tasks.

        Args:
            time_delta_s: Time step in seconds

        Returns:
            List of completed task results
        """
        self.current_time += time_delta_s
        completed = []

        # Process active tasks
        remaining_tasks = []
        for task in self.active_tasks:
            if task.completion_time and task.completion_time <= self.current_time:
                completed.append({
                    'task_id': task.task_id,
                    'device_id': task.device_id,
                    'latency_ms': (task.completion_time - task.arrival_time) * 1000,
                    'accuracy': task.accuracy,
                    'energy_j': task.energy_consumed
                })
            else:
                remaining_tasks.append(task)
        self.active_tasks = remaining_tasks

        return completed

    def get_observation(self) -> np.ndarray:
        """
        Get observation vector for this server.

        Returns:
            Observation array containing:
            - Queue length (normalized)
            - CPU utilization
            - Memory utilization
            - Number of active tasks (normalized)
            - Average queue wait time (normalized)
        """
        queue_length_norm = len(self.task_queue) / max(self.task_queue.maxlen, 1)
        active_tasks_norm = len(self.active_tasks) / max(self.config.max_concurrent_tasks, 1)

        # Estimate average wait time
        avg_wait = len(self.task_queue) * 10 / 100  # Normalized to 100ms max

        obs = np.array([
            queue_length_norm,
            self.cpu_utilization,
            self.memory_utilization,
            active_tasks_norm,
            min(avg_wait, 1.0)
        ], dtype=np.float32)
        # Sanitize observation
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
        obs = np.clip(obs, -10.0, 10.0)
        return obs

    def get_info(self) -> Dict[str, Any]:
        """Get server information dictionary."""
        return {
            'server_id': self.server_id,
            'queue_length': len(self.task_queue),
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'active_tasks': len(self.active_tasks),
            'total_processed': self.total_tasks_processed,
            'total_dropped': self.total_tasks_dropped,
            'total_energy': self.total_energy_consumed,
            'avg_latency': self.total_latency / max(1, self.total_tasks_processed)
        }


class CloudServer:
    """
    Simple cloud server simulation for cloud offloading option.

    Assumes virtually infinite resources but higher base latency.
    """

    def __init__(self, base_latency_ms: float = 50.0, latency_variance_ms: float = 20.0):
        """
        Initialize cloud server.

        Args:
            base_latency_ms: Base network latency to cloud
            latency_variance_ms: Variance in network latency
        """
        self.base_latency_ms = base_latency_ms
        self.latency_variance_ms = latency_variance_ms

        # Very high compute capacity (essentially infinite)
        self.compute_capacity = 1e12  # cycles per second

    def process_task(self, task: Task, model_accuracy: float) -> Dict[str, float]:
        """
        Process task on cloud.

        Args:
            task: Task to process
            model_accuracy: Base accuracy of perception model

        Returns:
            Processing results
        """
        # Network latency (round trip)
        network_latency = (
            self.base_latency_ms +
            np.random.uniform(-self.latency_variance_ms, self.latency_variance_ms)
        )

        # Compute time (very fast on cloud)
        compute_time = task.compute_cycles / self.compute_capacity * 1000  # ms

        # Total latency
        total_latency = network_latency * 2 + compute_time  # Round trip

        # Energy for transmission (simplified)
        transmission_energy = task.data_size_bits * 1e-9  # 1 nJ per bit

        return {
            'latency_ms': total_latency,
            'energy_j': transmission_energy,
            'accuracy': model_accuracy,  # Full accuracy on cloud
            'deadline_violated': total_latency > task.deadline_ms,
            'network_latency_ms': network_latency * 2,
            'compute_time_ms': compute_time
        }
