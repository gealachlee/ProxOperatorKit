#!/usr/bin/env python3
"""
测试动态配置文件路径的完整示例
"""

from pathlib import Path
from plot_runner_new import PlotConfig, get_plot_config


def create_test_configs():
    """创建测试用的配置文件"""
    
    # 创建高优先级JSON配置
    high_priority_config = {
        "plot_width": 2000,
        "plot_height": 1200,
        "font_size": 20,
        "color_theme": "high_priority_theme",
        "dpi": 300
    }
    
    # 创建中优先级JSON配置
    medium_priority_config = {
        "plot_width": 1600,
        "plot_height": 1000,
        "font_size": 16,
        "color_theme": "medium_priority_theme",
        "background_color": "#f5f5f5"
    }
    
    # 创建低优先级JSON配置
    low_priority_config = {
        "plot_width": 1200,
        "plot_height": 800,
        "font_size": 14,
        "color_theme": "low_priority_theme",
        "line_color": "#ff0000",
        "save_format": "pdf"
    }
    
    import json
    
    with open('high_priority.json', 'w', encoding='utf-8') as f:
        json.dump(high_priority_config, f, indent=2, ensure_ascii=False)
    
    with open('medium_priority.json', 'w', encoding='utf-8') as f:
        json.dump(medium_priority_config, f, indent=2, ensure_ascii=False)
    
    with open('low_priority.json', 'w', encoding='utf-8') as f:
        json.dump(low_priority_config, f, indent=2, ensure_ascii=False)
    
    print("✅ 测试配置文件已创建")


def test_priority_system():
    """测试配置文件优先级系统"""
    
    print("=== 测试配置文件优先级系统 ===")
    
    # 测试1: 只有高优先级文件存在
    config1 = PlotConfig(json_files=['high_priority.json', 'nonexistent1.json', 'nonexistent2.json'])
    print("\n📋 测试1: 只使用第一个存在的文件")
    print(f"plot_width: {config1.plot_width} (应该是2000)")
    print(f"color_theme: {config1.color_theme} (应该是high_priority_theme)")
    
    # 测试2: 多个文件存在，使用第一个
    config2 = PlotConfig(json_files=['high_priority.json', 'medium_priority.json', 'low_priority.json'])
    print("\n📋 测试2: 多个文件存在，使用第一个")
    print(f"plot_width: {config2.plot_width} (应该是2000，来自high_priority.json)")
    print(f"color_theme: {config2.color_theme} (应该是high_priority_theme)")
    
    # 测试3: 第一个文件不存在，使用第二个
    config3 = PlotConfig(json_files=['nonexistent.json', 'medium_priority.json', 'low_priority.json'])
    print("\n📋 测试3: 第一个文件不存在，使用第二个")
    print(f"plot_width: {config3.plot_width} (应该是1600，来自medium_priority.json)")
    print(f"color_theme: {config3.color_theme} (应该是medium_priority_theme)")


def test_mixed_sources():
    """测试混合配置源"""
    
    print("\n=== 测试混合配置源 ===")
    
    # 测试: 直接参数 + 配置文件
    config = PlotConfig(
        json_files=['medium_priority.json'],
        plot_width=3000,  # 直接参数，应该覆盖文件中的值
        font_size=25      # 直接参数，应该覆盖文件中的值
    )
    
    print("📋 直接参数 + 配置文件:")
    print(f"plot_width: {config.plot_width} (应该是3000，来自直接参数)")
    print(f"font_size: {config.font_size} (应该是25，来自直接参数)")
    print(f"background_color: {config.background_color} (应该是#f5f5f5，来自配置文件)")
    print(f"line_color: {config.line_color} (应该是#000000，来自默认值)")


def test_factory_function():
    """测试工厂函数"""
    
    print("\n=== 测试工厂函数 ===")
    
    config = get_plot_config(
        json_files=['low_priority.json'],
        yaml_files=['config.yaml'],
        plot_height=1500,
        dpi=400
    )
    
    print("📋 使用工厂函数:")
    print(f"plot_width: {config.plot_width} (来自配置文件)")
    print(f"plot_height: {config.plot_height} (来自直接参数)")
    print(f"dpi: {config.dpi} (来自直接参数)")
    print(f"save_format: {config.save_format} (来自配置文件)")


def test_path_variations():
    """测试不同路径格式"""
    
    print("\n=== 测试不同路径格式 ===")
    
    # 测试1: 字符串路径
    config1 = PlotConfig(json_files='high_priority.json')
    print("📋 字符串路径:")
    print(f"JSON files: {config1.__class__._json_files}")
    
    # 测试2: 列表路径
    config2 = PlotConfig(json_files=['high_priority.json', 'medium_priority.json'])
    print("\n📋 列表路径:")
    print(f"JSON files: {config2.__class__._json_files}")
    
    # 测试3: Path对象路径
    config3 = PlotConfig(json_files=[Path('high_priority.json'), 'medium_priority.json'])
    print("\n📋 混合Path对象和字符串:")
    print(f"JSON files: {config3.__class__._json_files}")


def cleanup_test_files():
    """清理测试文件"""
    test_files = ['high_priority.json', 'medium_priority.json', 'low_priority.json']
    for file in test_files:
        try:
            Path(file).unlink()
            print(f"🗑️ 已删除: {file}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    try:
        create_test_configs()
        test_priority_system()
        test_mixed_sources()
        test_factory_function()
        test_path_variations()
    finally:
        cleanup_test_files()
        print("\n✅ 所有测试完成！")