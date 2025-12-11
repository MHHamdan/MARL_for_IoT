"""
Logging Utilities for MARL-IoTP

Provides logging functionality for training experiments including
console logging, file logging, and TensorBoard integration.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Main logger for training experiments.

    Handles:
    - Console output
    - File logging
    - JSON metrics storage
    - TensorBoard integration (optional)
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
            log_level: Logging level
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directories
        self.log_dir = Path(log_dir) / f"{experiment_name}_{self.timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logger
        self._setup_file_logger(log_level)

        # Metrics storage
        self.metrics_history: Dict[str, List[float]] = {}
        self.episode_history: List[Dict[str, Any]] = []

        # TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        self.logger.info(f"Logger initialized for experiment: {experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")

    def _setup_file_logger(self, log_level: int):
        """Setup file and console logging."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(log_level)

        # Clear existing handlers
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / "training.log"
        )
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    def log(self, message: str, level: int = logging.INFO):
        """Log a message."""
        self.logger.log(level, message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def log_episode(
        self,
        episode: int,
        reward: float,
        metrics: Dict[str, float],
        prefix: str = ""
    ):
        """
        Log episode results.

        Args:
            episode: Episode number
            reward: Total episode reward
            metrics: Dictionary of metrics
            prefix: Optional prefix for metric names
        """
        # Store metrics
        episode_data = {
            'episode': episode,
            'reward': reward,
            **metrics
        }
        self.episode_history.append(episode_data)

        # Update history
        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            if full_key not in self.metrics_history:
                self.metrics_history[full_key] = []
            self.metrics_history[full_key].append(value)

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar(f"{prefix}reward", reward, episode)
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}{key}", value, episode)

        # Console log (every N episodes)
        if episode % 100 == 0 or episode < 10:
            metrics_str = ", ".join([
                f"{k}: {v:.4f}" for k, v in metrics.items()
            ])
            self.logger.info(
                f"Episode {episode} | Reward: {reward:.2f} | {metrics_str}"
            )

    def log_training(
        self,
        train_info: Dict[str, float],
        step: int
    ):
        """
        Log training step information.

        Args:
            train_info: Dictionary of training metrics
            step: Training step
        """
        if self.writer:
            for key, value in train_info.items():
                self.writer.add_scalar(f"train/{key}", value, step)

    def log_evaluation(
        self,
        eval_metrics: Dict[str, float],
        episode: int
    ):
        """
        Log evaluation results.

        Args:
            eval_metrics: Evaluation metrics
            episode: Episode number
        """
        if self.writer:
            for key, value in eval_metrics.items():
                self.writer.add_scalar(f"eval/{key}", value, episode)

        metrics_str = ", ".join([
            f"{k}: {v:.4f}" for k, v in eval_metrics.items()
        ])
        self.logger.info(f"Evaluation at episode {episode}: {metrics_str}")

    def log_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info(f"Configuration saved to {config_path}")

    def save_metrics(self):
        """Save metrics history to file."""
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'history': self.metrics_history,
                'episodes': self.episode_history
            }, f, indent=2)

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        import numpy as np

        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]
                }
        return summary

    def close(self):
        """Close logger and save final metrics."""
        self.save_metrics()
        if self.writer:
            self.writer.close()
        self.logger.info("Logger closed")


class TensorBoardLogger:
    """
    Standalone TensorBoard logger for simpler use cases.
    """

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available")

        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value."""
        step = step if step is not None else self.step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log multiple scalar values."""
        step = step if step is not None else self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
        self,
        tag: str,
        values,
        step: Optional[int] = None
    ):
        """Log histogram."""
        step = step if step is not None else self.step
        self.writer.add_histogram(tag, values, step)

    def log_image(
        self,
        tag: str,
        img_tensor,
        step: Optional[int] = None
    ):
        """Log image."""
        step = step if step is not None else self.step
        self.writer.add_image(tag, img_tensor, step)

    def log_figure(
        self,
        tag: str,
        figure,
        step: Optional[int] = None
    ):
        """Log matplotlib figure."""
        step = step if step is not None else self.step
        self.writer.add_figure(tag, figure, step)

    def increment_step(self):
        """Increment global step."""
        self.step += 1

    def close(self):
        """Close writer."""
        self.writer.close()


class WandbLogger:
    """
    Weights & Biases logger (optional integration).
    """

    def __init__(
        self,
        project: str,
        experiment_name: str,
        config: Dict[str, Any]
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            experiment_name: Experiment name
            config: Configuration to log
        """
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                name=experiment_name,
                config=config
            )
            self.available = True
        except ImportError:
            self.available = False
            print("wandb not available, logging disabled")

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if self.available:
            self.wandb.log(data, step=step)

    def log_artifact(self, name: str, artifact_type: str, path: str):
        """Log artifact."""
        if self.available:
            artifact = self.wandb.Artifact(name, type=artifact_type)
            artifact.add_file(path)
            self.run.log_artifact(artifact)

    def finish(self):
        """Finish run."""
        if self.available:
            self.wandb.finish()
