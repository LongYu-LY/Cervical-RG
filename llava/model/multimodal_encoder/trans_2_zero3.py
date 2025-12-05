import torch
import deepspeed
from deepspeed.utils import logger
from transformers import AutoConfig

# 自定义模型类（需与您的 ViT3DTower 保持一致）
from vit import ViT3DTower, ViT  # 替换为实际模块路径

def convert_zero2_to_zero3(checkpoint_path, output_dir, model_config):
    """将 ZeRO-2 的 checkpoint 转换为 ZeRO-3 兼容格式"""
    
    # 步骤1：初始化原始模型
    model = ViT3DTower(model_config)
    
    # 步骤2：配置 ZeRO-2 环境
    ds_config = {
        "train_batch_size": 32,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "train_micro_batch_size_per_gpu": "auto"
    }
    
    # 步骤3：DeepSpeed 初始化（ZeRO-2）
    model.vision_tower, optimizer, _, _ = deepspeed.initialize(
        model=model.vision_tower,
        config_params=ds_config
    )
    
    # 步骤4：加载 ZeRO-2 checkpoint
    model.vision_tower.load_checkpoint(checkpoint_path)
    
    # 步骤5：聚合参数到 CPU
    with deepspeed.zero.GatheredParameters(model.vision_tower.parameters()):
        state_dict = model.vision_tower.state_dict()
    
    # 步骤6：保存完整参数
    torch.save(state_dict, "/home/zwding/ly/hf_hub/Cervice_CLIP/modified_model.bin")
    
    # 步骤7：重新初始化模型（避免对象复用）
    del model
    model = ViT3DTower(model_config)
    
    # 步骤8：配置 ZeRO-3
    ds_config["zero_optimization"] = {
        "stage": 3,
        "param_persistence_threshold": 1e5,
        "offload_param": {"device": "none"}
    }
    
    # 步骤9：DeepSpeed 初始化（ZeRO-3）
    model.vision_tower, optimizer, _, _ = deepspeed.initialize(
        model=model.vision_tower,
        config_params=ds_config
    )
    
    # 步骤10：加载完整参数并保存为 ZeRO-3 格式
    model.vision_tower.load_state_dict(torch.load("/home/zwding/ly/hf_hub/Cervice_CLIP/modified_model.bin"))
    model.vision_tower.save_checkpoint(output_dir)
    logger.info(f"ZeRO-3 checkpoint saved to {output_dir}")

class ViT3DTowerZero3(ViT3DTower):
    """支持 ZeRO-3 的改进版本"""
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.vision_tower_name} already loaded, skipping.")
            return
        
        # 步骤1：初始化 DeepSpeed 引擎
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "param_persistence_threshold": 1e5,
                "offload_param": {"device": "none"}
            },
            "train_micro_batch_size_per_gpu": "auto"
        }
        
        # 步骤2：DeepSpeed 初始化
        self.vision_tower, _, _, _ = deepspeed.initialize(
            model=self.vision_tower,
            config_params=ds_config
        )
        
        # 步骤3：加载 ZeRO-3 分片 checkpoint
        self.vision_tower.load_checkpoint(self.vision_tower_name)
        self.is_loaded = True
        print(f"Loaded ZeRO-3 checkpoint from {self.vision_tower_name}")

# 使用示例
if __name__ == "__main__":
    # 定义模型配置（需与您的实际配置一致）
    class Config:
        image_channel = 1
        image_size = (96, 96, 96)
        patch_size = (16, 16, 16)
        mm_vision_select_layer = -1
        mm_vision_select_feature = "patch"
        vision_tower_name = "/home/zwding/ly/hf_hub/Cervice_CLIP/modified_model.bin"
    
    config = Config()
    
    # Step 1: 转换 checkpoint
    convert_zero2_to_zero3(
        checkpoint_path=config.vision_tower_name,
        output_dir="/home/zwding/ly/hf_hub/Cervice_CLIP_zero3",
        model_config=config
    )
    
    # Step 2: 使用转换后的 checkpoint
    config.vision_tower_name = "/home/zwding/ly/hf_hub/Cervice_CLIP_zero3"
    model = ViT3DTowerZero3(config)
    model.load_model()
    
    # 测试推理
    dummy_input = torch.randn(2, 1, 96, 96, 96)  # 批量输入
    features = model(dummy_input)
    print(f"Output features shape: {features.shape}")