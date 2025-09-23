from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator, field_validator
from pydantic_settings import BaseSettings

__all__ = ["Settings"]


class DistributionType(str, Enum):
    """Supported probability distributions."""
    NORMAL = "normal"
    LAPLACE = "laplace"
    RAND = 'rand'


class ObjectiveType(str, Enum):
    """Supported objective functions."""
    REPEAT_NMSE = "repeat_nmse"
    NMSE = "nmse"
    RELATIVE = "relative"
    SUCCESS_TIMES = "success_times"


class SeparableSparsityMode(str, Enum):
    """Modes for separable sparsity handling."""
    PERCENTAGE = "percentage"
    BINOMIAL = "binomial"
    TOTAL_PERCENTAGE = "total_percentage"


class NoiseConfig(BaseModel):
    """Configuration for noise parameters."""
    loc: float = Field(default=0.0, description="Noise location parameter")
    scale: float = Field(default=1.0, ge=0.0, description="Noise scale parameter")
    sig: float = Field(default=0.01, ge=0.0, description="Noise standard deviation")
    distribution: DistributionType = Field(default=DistributionType.NORMAL)
    dist: DistributionType = Field(default=DistributionType.NORMAL,
                                   description="Alias for distribution, for backward compatibility")


class JointSparseConfig(BaseModel):
    """Configuration for joint sparsity settings."""
    mode: SeparableSparsityMode = Field(default=SeparableSparsityMode.BINOMIAL)
    is_joint_sparse: bool = Field(default=False, description="Alias for enabled, for backward compatibility")
    p: float = Field(default=0.5, ge=0.0, le=1.0, description="Alias for probability, for backward compatibility")
#
#
# class DatasetConfig(BaseModel):
#     """Configuration for dataset generation."""
#     m: int = Field(default=256, gt=0, description="Number of rows in sensing matrix")
#     n: int = Field(default=1024, gt=0, description="Number of columns in sensing matrix")
#     data_size: int = Field(default=16, gt=0, description="Number of samples")
#     sparsity: int = Field(default=8, gt=0, description="Group sparsity level")
#     group_length: int = Field(default=16, gt=0, description="Length of each group")
#     distribution: DistributionType = Field(default=DistributionType.NORMAL)
#     seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
#     noise: Optional[NoiseConfig] = Field(default_factory=NoiseConfig)
#     joint_sparse: JointSparseConfig = Field(default_factory=JointSparseConfig)
#     gLen: int = Field(default=16, gt=0, description="Alias for group_length, for backward compatibility")
#     dist: str = Field(default="normal", description="Alias for distribution, for backward compatibility")
#
#     @field_validator('n')
#     def validate_n_divisible_by_group_length(cls, v, values):
#         if 'group_length' in values and v % values['group_length'] != 0:
#             raise ValueError(f"n ({v}) must be divisible by group_length ({values['group_length']})")
#         return v


# class OptimizationConfig(BaseModel):
#     """Configuration for optimization algorithms."""
#     max_iterations: int = Field(default=1000, gt=0, description="Maximum number of iterations")
#     tau: float = Field(default=0.1, gt=0.0, description="Regularization parameter")
#     tolerance: float = Field(default=1e-6, gt=0.0, description="Convergence tolerance")
#     objective: ObjectiveType = Field(default=ObjectiveType.REPEAT_NMSE)
#     K: int = Field(default=1000, gt=0, description="Alias for max_iterations, for backward compatibility")
#     objective_alias: str = Field(default="Repeat NMSE", description="Alias for objective, for backward compatibility")


# class LoggingConfig(BaseModel):
#     """Configuration for logging system."""
#     level: str = Field(default="INFO", description="Logging level")
#     format: str = Field(default="json", description="Log format (json/text)")
#     file_path: Optional[Path] = Field(default=None, description="Log file path")
#     console_output: bool = Field(default=True, description="Enable console output")
#     logger: Optional[Any] = Field(default=None, description="Logger instance, for backward compatibility")

    # def init_log(self, opts: Any) -> None:
    #     """Initialize logger based on configuration."""
    #     if not self.file_path:
    #         self.file_path = Path("output/output.log")
    #     if not self.file_path.parent.exists():
    #         self.file_path.parent.mkdir(parents=True)
    #     self.logger = utils.setup_logger(str(self.file_path))
    #     self.logger(f"Checkpoints will be saved to directory `{self.file_path.parent}`")
    #     self.logger(f"Log file for training will be saved to file `{self.file_path}`")
    #     self.logger(f"Using tau: {opts.tau}")

#
# class ExperimentConfig(BaseModel):
#     """Configuration for experiment execution."""
#     name: str = Field(description="Experiment name")
#     description: Optional[str] = Field(default=None, description="Experiment description")
#     output_dir: Path = Field(default=Path("output"), description="Output directory")
#     save_plots: bool = Field(default=True, description="Save generated plots")
#     save_results: bool = Field(default=True, description="Save experiment results")
#     parallel_execution: bool = Field(default=False, description="Enable parallel execution")
#     max_workers: Optional[int] = Field(default=None, description="Maximum number of workers")
#     save_dir: Path = Field(default=Path("output"), description="Alias for output_dir, for backward compatibility")


#
# class ModelConfig(BaseModel):
#     """Configuration for optimization models."""
#     algorithms: List[str] = Field(default=["FISTA", "ISTA"], description="Algorithms to use")
#     proximal_operators: List[str] = Field(default=["ProxL2_1", "ProxL2_2over3"],
#                                           description="Proximal operators to use")
#     custom_params: Dict[str, Any] = Field(default_factory=dict, description="Custom model parameters")
#     algorithm: str = Field(default="FISTA", description="Alias for algorithms[0], for backward compatibility")
#

class Settings(BaseSettings):
    """Main application settings with environment variable support."""

    # Core settings (aligned with Opts)
    description: str = Field(default="Configurations for Group Sparse experiment.",
                             description="Experiment description")
    K: int = Field(default=1000, gt=0, description="Total iterations (alias for max_iterations)")
    tau: float = Field(default=0.1, gt=0.0, description="Regularization parameter")
    m: int = Field(default=256, gt=0, description="Number of rows in sensing matrix")
    n: int = Field(default=1024, gt=0, description="Number of columns in sensing matrix")
    data_size: int = Field(default=16, gt=0, description="Number of samples")
    sparsity: int = Field(default=8, gt=0, description="Group sparsity level")
    gLen: int = Field(default=16, gt=0, description="Length of each group")
    dist: str = Field(default="normal", description="Distribution of entries in the matrix A")
    data_seed: int = Field(default=6, description="Random seed for data generation")
    save_dir: Path = Field(default=Path("output"), description="Output directory")
    plot_figs: bool = Field(default=True, description="Save generated plots")
    objective: str = Field(default="Repeat NMSE", description="Loss function")
    logger: Optional[Any] = Field(default=None, description="Logger instance")

    # Nested configurations (aligned with Opts)
    noise_params: NoiseConfig = Field(default_factory=NoiseConfig, description="Noise parameters")
    joint_sparse_config: JointSparseConfig = Field(default_factory=JointSparseConfig,
                                                   description="Joint sparsity settings")
