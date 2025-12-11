"""
IoT Edge Computing Environment for Multi-Agent Reinforcement Learning

Main environment class implementing a Gymnasium-compatible multi-agent
environment for intelligent perception and resource orchestration.

Features:
- Multiple IoT devices with perception tasks (object detection simulation)
- Multiple edge servers with computation resources
- Wireless channel model with stochastic conditions
- Support for heterogeneous agents (perception + orchestration)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional

from envs.network_model import WirelessChannel, ChannelConfig, ChannelSimulator
from envs.edge_server import EdgeServer, ServerConfig, Task, CloudServer
from envs.perception_task import PerceptionTaskGenerator, IoTDevice


class IoTEdgeEnv(gym.Env):
    """
    IoT Edge Computing Environment for MARL.

    State Space (per device):
    - Task queue length
    - Current perception model index
    - Channel quality (SNR)
    - Battery level
    - CPU utilization

    Action Space:
    - Perception Agent: [model_selection (discrete), frame_rate (continuous)]
    - Orchestration Agent: [offload_decision (discrete), resource_allocation (continuous)]

    Reward:
    - Composite: -alpha*latency - beta*energy + gamma*accuracy
    - With penalties for deadline violations and dropped tasks
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, config: dict):
        """
        Initialize the IoT Edge environment.

        Args:
            config: Dictionary containing:
                - num_iot_devices: Number of IoT devices (10-50)
                - num_edge_servers: Number of edge servers (3-5)
                - num_perception_models: Available perception models (3-5)
                - max_queue_length: Maximum task queue per device
                - episode_length: Steps per episode
                - reward_weights: {'latency': alpha, 'energy': beta, 'accuracy': gamma}
        """
        super().__init__()
        self.config = config

        # Environment parameters
        self.num_devices = config.get('num_iot_devices', 20)
        self.num_servers = config.get('num_edge_servers', 3)
        self.num_models = config.get('num_perception_models', 4)
        self.max_queue_length = config.get('max_queue_length', 10)
        self.episode_length = config.get('episode_length', 200)
        self.devices_per_server = config.get(
            'devices_per_server',
            int(np.ceil(self.num_devices / self.num_servers))
        )

        # Reward weights
        reward_weights = config.get('reward_weights', {})
        self.alpha = reward_weights.get('latency', 0.4)
        self.beta = reward_weights.get('energy', 0.3)
        self.gamma = reward_weights.get('accuracy', 0.3)
        self.deadline_penalty = config.get('deadline_penalty', 10.0)

        # Observation dimensions
        self.perception_obs_dim = config.get('perception_obs_dim', 16)
        self.orchestration_obs_dim = config.get('orchestration_obs_dim', 32)
        self.message_dim = config.get('message_dim', 8)

        # Initialize components
        self._init_channel()
        self._init_devices()
        self._init_servers()
        self._init_perception()

        # Define action and observation spaces
        self._define_spaces()

        # Episode tracking
        self.current_step = 0
        self.episode_rewards = []

        # Device-server assignment
        self._assign_devices_to_servers()

    def _init_channel(self):
        """Initialize wireless channel model."""
        channel_config = ChannelConfig(
            bandwidth_mhz=self.config.get('bandwidth_mhz', 20),
            noise_power_dbm=self.config.get('noise_power_dbm', -100),
            path_loss_exponent=self.config.get('path_loss_exponent', 3.5)
        )
        self.channel = WirelessChannel(channel_config)
        self.channel_simulator = ChannelSimulator(
            channel_config,
            doppler_freq_hz=10.0,
            sample_interval_s=0.1
        )

    def _init_devices(self):
        """Initialize IoT devices."""
        device_config = self.config.get('iot_device', {})
        self.devices: List[IoTDevice] = []

        for i in range(self.num_devices):
            device = IoTDevice(
                device_id=i,
                cpu_frequency_ghz=device_config.get('cpu_frequency_ghz', 1.5),
                battery_capacity_mah=device_config.get('battery_capacity_mah', 5000),
                transmit_power_dbm=device_config.get('transmit_power_dbm', 20)
            )
            # Random positions in 100m x 100m area
            device.position = np.random.uniform(0, 100, size=2)
            self.devices.append(device)

    def _init_servers(self):
        """Initialize edge servers."""
        server_config_dict = self.config.get('edge_server', {})
        server_config = ServerConfig(
            cpu_frequency_ghz=server_config_dict.get('cpu_frequency_ghz', 3.0),
            num_cores=server_config_dict.get('num_cores', 8),
            memory_gb=server_config_dict.get('memory_gb', 16),
            max_concurrent_tasks=server_config_dict.get('max_concurrent_tasks', 10)
        )

        self.servers: List[EdgeServer] = []
        self.server_positions = np.zeros((self.num_servers, 2))

        for i in range(self.num_servers):
            server = EdgeServer(server_id=i, config=server_config)
            self.servers.append(server)
            # Distribute servers evenly in the area
            self.server_positions[i] = np.array([
                50 + 30 * np.cos(2 * np.pi * i / self.num_servers),
                50 + 30 * np.sin(2 * np.pi * i / self.num_servers)
            ])

        # Cloud server
        cloud_config = self.config.get('cloud', {})
        self.cloud = CloudServer(
            base_latency_ms=cloud_config.get('latency_base_ms', 50),
            latency_variance_ms=cloud_config.get('latency_variance_ms', 20)
        )

    def _init_perception(self):
        """Initialize perception task generator."""
        model_configs = self.config.get('perception_models', [
            {'name': 'MobileNet-Tiny', 'accuracy': 0.65, 'latency_factor': 0.3,
             'energy_factor': 0.2, 'flops_million': 50},
            {'name': 'MobileNet-Small', 'accuracy': 0.75, 'latency_factor': 0.5,
             'energy_factor': 0.4, 'flops_million': 150},
            {'name': 'MobileNet-Large', 'accuracy': 0.85, 'latency_factor': 0.8,
             'energy_factor': 0.7, 'flops_million': 300},
            {'name': 'ResNet-Edge', 'accuracy': 0.92, 'latency_factor': 1.0,
             'energy_factor': 1.0, 'flops_million': 500},
        ])
        self.task_generator = PerceptionTaskGenerator(model_configs)

    def _assign_devices_to_servers(self):
        """Assign devices to nearest servers."""
        self.device_to_server = {}
        for device in self.devices:
            distances = np.linalg.norm(
                self.server_positions - device.position, axis=1
            )
            self.device_to_server[device.device_id] = np.argmin(distances)
            self.servers[self.device_to_server[device.device_id]].connected_devices.append(
                device.device_id
            )

    def _define_spaces(self):
        """Define observation and action spaces for all agents."""
        # Perception agent spaces
        self.perception_observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.perception_obs_dim,),
            dtype=np.float32
        )

        self.perception_action_space = spaces.Dict({
            'model_selection': spaces.Discrete(self.num_models),
            'frame_rate': spaces.Box(low=0.1, high=1.0, shape=(1,), dtype=np.float32)
        })

        # Orchestration agent spaces
        self.orchestration_observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.orchestration_obs_dim,),
            dtype=np.float32
        )

        # Per-device offload decision (0=local, 1=edge, 2=cloud)
        self.orchestration_action_space = spaces.Dict({
            'offload_decisions': spaces.MultiDiscrete(
                [3] * self.devices_per_server
            ),
            'resource_allocation': spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            'bandwidth_allocation': spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        })

        # Combined spaces for gym compatibility
        self.observation_space = spaces.Dict({
            'perception': spaces.Tuple([
                self.perception_observation_space
                for _ in range(self.num_devices)
            ]),
            'orchestration': spaces.Tuple([
                self.orchestration_observation_space
                for _ in range(self.num_servers)
            ])
        })

        self.action_space = spaces.Dict({
            'perception': spaces.Tuple([
                self.perception_action_space
                for _ in range(self.num_devices)
            ]),
            'orchestration': spaces.Tuple([
                self.orchestration_action_space
                for _ in range(self.num_servers)
            ])
        })

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment and return initial observations for all agents.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observations, info)
        """
        super().reset(seed=seed)

        # Reset all components
        for device in self.devices:
            device.reset()
        for server in self.servers:
            server.reset()
        self.channel_simulator.reset()

        # Reset episode tracking
        self.current_step = 0
        self.episode_rewards = []

        # Update channel conditions
        self._update_channel_conditions()

        # Generate initial tasks
        self._generate_tasks()

        # Get observations
        observations = self._get_observations()

        info = {
            'num_devices': self.num_devices,
            'num_servers': self.num_servers,
            'episode_length': self.episode_length
        }

        return observations, info

    def step(
        self,
        actions: Dict
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Execute actions for all agents.

        Args:
            actions: Dict with keys 'perception' and 'orchestration'

        Returns:
            observations, rewards, terminated, truncated, infos (all as Dicts)
        """
        self.current_step += 1

        # Process perception agent actions
        perception_actions = actions.get('perception', [])
        self._apply_perception_actions(perception_actions)

        # Process orchestration agent actions
        orchestration_actions = actions.get('orchestration', [])
        results = self._apply_orchestration_actions(orchestration_actions)

        # Update channel conditions
        self._update_channel_conditions()

        # Generate new tasks
        self._generate_tasks()

        # Advance server simulation
        for server in self.servers:
            server.step(0.1)  # 100ms time step

        # Compute rewards
        rewards = self._compute_rewards(results)

        # Get new observations
        observations = self._get_observations()

        # Check termination
        terminated = False
        truncated = self.current_step >= self.episode_length

        # Collect info
        info = self._collect_info(results)

        return observations, rewards, terminated, truncated, info

    def _get_observations(self) -> Dict:
        """Get observations for all agents."""
        # Perception agent observations
        perception_obs = []
        for device in self.devices:
            obs = self._get_perception_observation(device)
            perception_obs.append(obs)

        # Orchestration agent observations
        orchestration_obs = []
        for server in self.servers:
            obs = self._get_orchestration_observation(server)
            orchestration_obs.append(obs)

        return {
            'perception': perception_obs,
            'orchestration': orchestration_obs
        }

    def _get_perception_observation(self, device: IoTDevice) -> np.ndarray:
        """
        Get observation for a perception agent.

        Observation contains:
        - Device state (battery, queue, current model)
        - Channel quality to connected server
        - Task statistics
        """
        server_idx = self.device_to_server[device.device_id]
        channel_quality = self.channel.get_channel_quality(
            device.device_id, server_idx
        )

        # Build observation vector
        obs = np.zeros(self.perception_obs_dim, dtype=np.float32)

        # Device state
        device_obs = device.get_observation()
        obs[:5] = device_obs

        # Channel quality
        obs[5] = channel_quality

        # Model specifications (normalized)
        current_model = self.task_generator.get_model(device.current_model_idx)
        obs[6] = current_model.accuracy
        obs[7] = current_model.latency_factor
        obs[8] = current_model.energy_factor

        # Queue statistics
        if device.pending_tasks:
            avg_deadline = np.mean([t['deadline_ms'] for t in device.pending_tasks])
            avg_complexity = np.mean([t['scene_complexity'] for t in device.pending_tasks])
        else:
            avg_deadline = 100.0
            avg_complexity = 1.0
        obs[9] = avg_deadline / 200.0  # Normalized
        obs[10] = avg_complexity / 2.0

        # Server state (limited info)
        server = self.servers[server_idx]
        obs[11] = server.cpu_utilization
        obs[12] = len(server.task_queue) / 100.0

        # Time step
        obs[13] = self.current_step / self.episode_length

        # Sanitize observation - replace NaN/Inf with safe values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -10.0, 10.0)

        return obs

    def _get_orchestration_observation(self, server: EdgeServer) -> np.ndarray:
        """
        Get observation for an orchestration agent.

        Observation contains:
        - Server state (resources, queues)
        - Connected device states
        - Channel conditions summary
        """
        obs = np.zeros(self.orchestration_obs_dim, dtype=np.float32)

        # Server state
        server_obs = server.get_observation()
        obs[:5] = server_obs

        # Connected devices summary
        connected_devices = [
            self.devices[d_id] for d_id in server.connected_devices
        ]

        if connected_devices:
            # Average battery level
            obs[5] = np.mean([d.battery_level for d in connected_devices])
            # Average queue length
            obs[6] = np.mean([
                len(d.pending_tasks) / d.max_queue_size
                for d in connected_devices
            ])
            # Total pending tasks
            obs[7] = sum([len(d.pending_tasks) for d in connected_devices]) / 50.0

        # Channel conditions
        avg_channel_quality = np.mean([
            self.channel.get_channel_quality(d.device_id, server.server_id)
            for d in connected_devices
        ]) if connected_devices else 0.5
        obs[8] = avg_channel_quality

        # Available resources
        resources = server.get_available_resources()
        obs[9] = resources['cpu']
        obs[10] = resources['memory']

        # Time step
        obs[11] = self.current_step / self.episode_length

        # Workload prediction (simple: based on recent tasks)
        obs[12] = server.total_tasks_processed / max(self.current_step, 1) / 10.0

        # Sanitize observation - replace NaN/Inf with safe values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -10.0, 10.0)

        return obs

    def _apply_perception_actions(self, actions: List):
        """Apply perception agent actions."""
        for i, device in enumerate(self.devices):
            if i < len(actions) and actions[i] is not None:
                action = actions[i]

                # Update model selection
                if isinstance(action, dict):
                    model_idx = action.get('model_selection', 0)
                    frame_rate = action.get('frame_rate', [1.0])[0]
                else:
                    # Handle array format
                    model_idx = int(action[0]) if len(action) > 0 else 0
                    frame_rate = float(action[1]) if len(action) > 1 else 1.0

                device.current_model_idx = np.clip(model_idx, 0, self.num_models - 1)
                device.current_frame_rate = np.clip(frame_rate, 0.1, 1.0)

    def _apply_orchestration_actions(self, actions: List) -> List[Dict]:
        """
        Apply orchestration agent actions and process tasks.

        Returns:
            List of processing results
        """
        all_results = []

        for server_idx, server in enumerate(self.servers):
            if server_idx >= len(actions) or actions[server_idx] is None:
                continue

            action = actions[server_idx]

            # Parse action
            if isinstance(action, dict):
                offload_decisions = action.get('offload_decisions', [])
                resource_alloc = action.get('resource_allocation', [0.5])[0]
                bandwidth_alloc = action.get('bandwidth_allocation', [0.5])[0]
            else:
                # Handle array format
                n_devices = len(server.connected_devices)
                offload_decisions = action[:n_devices].astype(int) if len(action) > 0 else []
                resource_alloc = float(action[n_devices]) if len(action) > n_devices else 0.5
                bandwidth_alloc = float(action[n_devices + 1]) if len(action) > n_devices + 1 else 0.5

            # Process tasks from connected devices
            for i, device_id in enumerate(server.connected_devices):
                device = self.devices[device_id]

                if not device.pending_tasks:
                    continue

                task_dict = device.pending_tasks.pop(0)
                offload = offload_decisions[i] if i < len(offload_decisions) else 1

                # Create Task object
                task = Task(
                    task_id=task_dict['task_id'],
                    device_id=device_id,
                    data_size_bits=task_dict['data_size_bits'],
                    compute_cycles=task_dict['compute_cycles'],
                    deadline_ms=task_dict['deadline_ms'],
                    perception_model_idx=task_dict['model_idx'],
                    arrival_time=0
                )

                # Process based on offload decision
                if offload == 0:  # Local processing
                    result = device.local_process(task_dict, self.task_generator)
                elif offload == 1:  # Edge processing
                    result = server.process_task(
                        task,
                        allocated_cpu_fraction=resource_alloc,
                        model_accuracy=task_dict['base_accuracy']
                    )
                    # Add transmission latency
                    distance = np.linalg.norm(
                        device.position - self.server_positions[server_idx]
                    )
                    tx_time = self.channel.compute_transmission_time(
                        task_dict['data_size_bits'],
                        device.transmit_power_dbm,
                        max(distance, 1.0),
                        bandwidth_fraction=bandwidth_alloc
                    ) * 1000  # Convert to ms
                    result['latency_ms'] += tx_time
                    result['transmission_latency_ms'] = tx_time
                else:  # Cloud processing
                    result = self.cloud.process_task(task, task_dict['base_accuracy'])

                result['device_id'] = device_id
                result['server_id'] = server_idx
                result['offload_decision'] = offload
                all_results.append(result)

        return all_results

    def _compute_rewards(self, results: List[Dict]) -> Dict:
        """Compute rewards for all agents."""
        # Initialize rewards
        perception_rewards = [0.0] * self.num_devices
        orchestration_rewards = [0.0] * self.num_servers

        for result in results:
            device_id = result['device_id']
            server_id = result['server_id']

            # Normalize metrics with bounds
            latency_ms = float(np.clip(result.get('latency_ms', 0), 0, 1000))
            energy_j = float(np.clip(result.get('energy_j', 0), 0, 10))
            accuracy = float(np.clip(result.get('accuracy', 0), 0, 1))

            latency_norm = latency_ms / 100.0  # 100ms baseline
            energy_norm = energy_j * 10.0  # Scale for visibility

            # Composite reward
            reward = (
                -self.alpha * latency_norm
                - self.beta * energy_norm
                + self.gamma * accuracy
            )

            # Deadline penalty
            if result.get('deadline_violated', False):
                reward -= self.deadline_penalty

            # Clip reward to prevent extreme values
            reward = float(np.clip(reward, -20.0, 10.0))

            # Assign to agents
            perception_rewards[device_id] += reward
            orchestration_rewards[server_id] += reward

        # Clip final rewards
        perception_rewards = [float(np.clip(r, -50.0, 50.0)) for r in perception_rewards]
        orchestration_rewards = [float(np.clip(r, -50.0, 50.0)) for r in orchestration_rewards]

        return {
            'perception': perception_rewards,
            'orchestration': orchestration_rewards,
            'total': sum(perception_rewards) + sum(orchestration_rewards)
        }

    def _update_channel_conditions(self):
        """Update wireless channel conditions."""
        device_positions = np.array([d.position for d in self.devices])
        self.channel.update_channel_state(
            self.num_devices,
            self.num_servers,
            device_positions,
            self.server_positions
        )

    def _generate_tasks(self):
        """Generate new perception tasks for devices."""
        for device in self.devices:
            # Generate task with some probability
            if np.random.random() < 0.3:  # 30% chance per step
                device.generate_task(
                    self.task_generator,
                    model_idx=device.current_model_idx,
                    frame_rate=device.current_frame_rate
                )

    def _collect_info(self, results: List[Dict]) -> Dict:
        """Collect episode information."""
        if not results:
            return {
                'avg_latency': 0,
                'avg_energy': 0,
                'avg_accuracy': 0,
                'deadline_violations': 0,
                'tasks_processed': 0
            }

        return {
            'avg_latency': np.mean([r['latency_ms'] for r in results]),
            'avg_energy': np.mean([r['energy_j'] for r in results]),
            'avg_accuracy': np.mean([r['accuracy'] for r in results]),
            'deadline_violations': sum([
                1 for r in results if r.get('deadline_violated', False)
            ]),
            'tasks_processed': len(results),
            'offload_distribution': {
                'local': sum(1 for r in results if r['offload_decision'] == 0),
                'edge': sum(1 for r in results if r['offload_decision'] == 1),
                'cloud': sum(1 for r in results if r['offload_decision'] == 2)
            }
        }

    def get_agent_messages(self) -> Dict[int, np.ndarray]:
        """
        Get messages from perception agents for orchestration agents.

        Returns:
            Dictionary mapping device_id to message vector
        """
        messages = {}
        for device in self.devices:
            # Encode message based on device state
            message = np.zeros(self.message_dim, dtype=np.float32)
            message[0] = device.battery_level
            message[1] = len(device.pending_tasks) / device.max_queue_size
            message[2] = device.current_model_idx / (self.num_models - 1)
            message[3] = device.current_frame_rate

            if device.pending_tasks:
                message[4] = device.pending_tasks[0]['deadline_ms'] / 200.0
                message[5] = device.pending_tasks[0]['scene_complexity'] / 2.0
            else:
                message[4] = 0.5
                message[5] = 0.5

            server_idx = self.device_to_server[device.device_id]
            message[6] = self.channel.get_channel_quality(
                device.device_id, server_idx
            )

            messages[device.device_id] = message

        return messages

    def render(self, mode: str = 'human'):
        """Render environment state (for debugging)."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Devices: {self.num_devices}, Servers: {self.num_servers}")

            for server in self.servers:
                info = server.get_info()
                print(f"Server {info['server_id']}: "
                      f"Queue={info['queue_length']}, "
                      f"CPU={info['cpu_utilization']:.2f}, "
                      f"Processed={info['total_processed']}")

    def close(self):
        """Clean up environment resources."""
        pass
