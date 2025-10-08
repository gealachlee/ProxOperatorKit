from pathlib import Path
from typing import List, Tuple, Type, Optional, Union

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import (
    PydanticBaseSettingsSource,
    EnvSettingsSource,
    DotEnvSettingsSource,
    JsonConfigSettingsSource,
    YamlConfigSettingsSource,
)
from pydantic import Field


class PlotConfig(BaseSettings):
    # 绘图尺寸配置
    plot_width: int = Field(default=800, description="绘图宽度", ge=100, le=5000)
    plot_height: int = Field(default=600, description="绘图高度", ge=100, le=5000)

    # is_save:bool = Field(default=True,description="是否保存图片")
    # save_dir:bool
    # 高级配置
    dpi: int = Field(default=600, description="分辨率DPI", ge=50, le=800)

    plot_color:list = Field(default=[],description="plot color")

    # 配置模型设置
    model_config = SettingsConfigDict(
        env_prefix='PLOT_',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    def __init__(self,
                 json_files: Optional[Union[str, List[str]]] = None,
                 yaml_files: Optional[Union[str, List[str]]] = None,
                 env_files: Optional[Union[str, List[str]]] = None,
                 **kwargs):
        """
        初始化配置，支持传入配置文件路径列表
        
        Args:
            json_files: JSON配置文件路径（单个文件或文件列表）
            yaml_files: YAML配置文件路径（单个文件或文件列表）
            env_files: .env文件路径（单个文件或文件列表）
            **kwargs: 其他配置参数
        """
        # 设置配置文件路径
        self.__class__._json_files = self._normalize_file_paths(json_files, [
            'config_demo.json',
            'config/config_demo.json',
            Path(__file__).parent / 'config_demo.json'
        ])

        self.__class__._yaml_files = self._normalize_file_paths(yaml_files, [
            'plot_config.yaml',
            'plot_config.yml',
            'config/plot_config.yaml',
            Path(__file__).parent / 'plot_config.yaml'
        ])

        self.__class__._env_files = self._normalize_file_paths(env_files, [
            '.env',
            'config/.env',
            Path(__file__).parent / '.env'
        ])

        super().__init__(**kwargs)

    @staticmethod
    def _normalize_file_paths(files: Optional[Union[str, List[str]]],
                              defaults: List[Union[str, Path]]) -> List[str]:
        """
        标准化文件路径列表
        
        Args:
            files: 用户提供的文件路径
            defaults: 默认文件路径列表
            
        Returns:
            标准化后的文件路径列表
        """
        if files is None:
            return [str(p) for p in defaults]
        elif isinstance(files, str):
            return [files]
        elif isinstance(files, list):
            return [str(f) for f in files]
        else:
            return [str(p) for p in defaults]

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: EnvSettingsSource,
            dotenv_settings: DotEnvSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        自定义配置源的优先级顺序
        优先级：初始化参数 > 环境变量 > .env文件 > JSON文件 > YAML文件 > 默认值
        """
        sources = [
            init_settings,  # 初始化参数（最高优先级）
            env_settings,  # 环境变量
        ]

        # 动态添加 .env 文件源
        if hasattr(cls, '_env_files'):
            for env_file in cls._env_files:
                if Path(env_file).exists():
                    sources.append(DotEnvSettingsSource(settings_cls, env_file=env_file))
                    break  # 只使用第一个存在的文件

        # 动态添加 JSON 文件源
        if hasattr(cls, '_json_files'):
            for json_file in cls._json_files:
                if Path(json_file).exists():
                    sources.append(JsonConfigSettingsSource(settings_cls, json_file=json_file))
                    break  # 只使用第一个存在的文件

        # 动态添加 YAML 文件源
        if hasattr(cls, '_yaml_files'):
            for yaml_file in cls._yaml_files:
                if Path(yaml_file).exists():
                    sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file))
                    break  # 只使用第一个存在的文件

        return tuple(sources)



    def print_config(self) -> None:
        """打印当前配置"""
        print("当前绘图配置:")
        for field_name, field_value in self.model_dump().items():
            print(f"  {field_name}: {field_value}")







