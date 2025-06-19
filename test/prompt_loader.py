#!/usr/bin/env python3
"""
Prompt加载器工具
从prompt_hub目录随机获取prompts
"""

import json
import random
import os
from typing import List, Optional

class PromptLoader:
    """从prompt_hub加载随机prompts的工具类"""
    
    def __init__(self, prompt_hub_path: str = "../prompt_hub"):
        self.prompt_hub_path = prompt_hub_path
        self.short_prompts: List[str] = []
        self.long_prompts: List[str] = []
        self._load_prompts()
    
    def _load_prompts(self):
        """加载所有prompt文件"""
        try:
            # 加载短prompts
            short_path = os.path.join(self.prompt_hub_path, "short_prompts.json")
            if os.path.exists(short_path):
                with open(short_path, 'r', encoding='utf-8') as f:
                    self.short_prompts = json.load(f)
                    print(f"✓ 加载了 {len(self.short_prompts)} 个短prompts")
            
            # 加载长prompts
            long_path = os.path.join(self.prompt_hub_path, "long_prompts.json")
            if os.path.exists(long_path):
                with open(long_path, 'r', encoding='utf-8') as f:
                    self.long_prompts = json.load(f)
                    print(f"✓ 加载了 {len(self.long_prompts)} 个长prompts")
                    
        except Exception as e:
            print(f"❌ 加载prompts失败: {e}")
    
    def get_random_prompt(self, prompt_type: str = "mixed") -> Optional[str]:
        """
        获取随机prompt
        
        Args:
            prompt_type: "short", "long", 或 "mixed"
        
        Returns:
            随机选择的prompt字符串
        """
        if prompt_type == "short" and self.short_prompts:
            return random.choice(self.short_prompts)
        elif prompt_type == "long" and self.long_prompts:
            return random.choice(self.long_prompts)
        elif prompt_type == "mixed":
            all_prompts = self.short_prompts + self.long_prompts
            if all_prompts:
                return random.choice(all_prompts)
        
        return None
    
    def get_random_prompts(self, count: int, prompt_type: str = "mixed") -> List[str]:
        """
        获取多个随机prompts
        
        Args:
            count: 需要的prompt数量
            prompt_type: "short", "long", 或 "mixed"
        
        Returns:
            随机prompts列表
        """
        prompts = []
        
        for _ in range(count):
            prompt = self.get_random_prompt(prompt_type)
            if prompt:
                prompts.append(prompt)
        
        return prompts
    
    def get_stats(self) -> dict:
        """获取prompts统计信息"""
        return {
            "short_prompts_count": len(self.short_prompts),
            "long_prompts_count": len(self.long_prompts),
            "total_prompts": len(self.short_prompts) + len(self.long_prompts)
        }

def test_prompt_loader():
    """测试prompt加载器"""
    print("=== 测试Prompt加载器 ===")
    
    loader = PromptLoader()
    
    # 显示统计信息
    stats = loader.get_stats()
    print(f"统计信息: {stats}")
    
    # 测试获取随机prompts
    print("\n=== 随机prompts测试 ===")
    
    # 获取一个短prompt
    short_prompt = loader.get_random_prompt("short")
    if short_prompt:
        print(f"随机短prompt: {short_prompt[:100]}...")
    
    # 获取一个长prompt
    long_prompt = loader.get_random_prompt("long")
    if long_prompt:
        print(f"随机长prompt: {long_prompt[:100]}...")
    
    # 获取混合prompts
    mixed_prompts = loader.get_random_prompts(3, "mixed")
    print(f"\n获取了 {len(mixed_prompts)} 个混合prompts")
    for i, prompt in enumerate(mixed_prompts, 1):
        print(f"{i}. {prompt[:80]}...")

if __name__ == "__main__":
    test_prompt_loader() 