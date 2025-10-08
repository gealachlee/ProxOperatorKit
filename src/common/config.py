from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, validator, field_validator
from pydantic_settings import BaseSettings

__all__ = ["Settings", "DistributionType", "ObjectiveType", "SeparableSparsityMode", "NoiseConfig", "JointSparseConfig", "LogConfig"]


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


class LogConfig(BaseModel):
    file_dir:Path = Field(default=Path('./'),description='Logging path')
    file_name: str = Field(default='experiment.log', description='Logging file name')
    is_debug:bool = Field(default=True,description='Configurations for generate output file or not')
    

class Settings(BaseSettings):
    """Main application settings with environment variable support."""

    # Core settings (aligned with ExpContainer)
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
    save_dir: Path = Field(default_factory=Path, description="Output directory")
    plot_figs: bool = Field(default=True, description="Save generated plots")
    objective: str = Field(default="Repeat NMSE", description="Loss function")
    log_config: LogConfig = Field(default_factory=LogConfig,
                                  description="Logging configurations")
    # Nested configurations (aligned with ExpContainer)
    noise_params: NoiseConfig = Field(default_factory=NoiseConfig, description="Noise parameters")
    joint_sparse_config: JointSparseConfig = Field(default_factory=JointSparseConfig,
                                                   description="Joint sparsity settings")

    def desc(self):
        """Show description of the experiment."""
        js = self.model_json_schema().get('properties')
        for key,value in js.items():
            print(f"{key}: {value.get('type')}, {value.get('description')}")