import torch
from Jamba.configuration_jamba import JambaConfig
from Jamba.modeling_jamba import JambaForCausalLM
from transformers import AutoTokenizer
import argparse

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--original_model_path", type=str, help="Path to the original dense model")
parser.add_argument("--moe_model_path", type=str, help="Path to the converted MoE model")
args = parser.parse_args()

# 加载原始模型和MoE模型
tokenizer = AutoTokenizer.from_pretrained(args.original_model_path, trust_remote_code=True)
original_config = JambaConfig.from_pretrained(args.original_model_path, trust_remote_code=True).to_dict()
moe_config = JambaConfig.from_pretrained(args.moe_model_path, trust_remote_code=True).to_dict()

dense_model = JambaForCausalLM.from_pretrained(args.original_model_path, trust_remote_code=True)
moe_model = JambaForCausalLM.from_pretrained(args.moe_model_path, trust_remote_code=True)

# 参数检查函数
def check_common_parameters(dense_model, moe_model):
    """检查非MoE层的参数一致性"""
    dense_params = dict(dense_model.named_parameters())
    moe_params = dict(moe_model.named_parameters())
    
    print("="*40)
    print("Common Parameters Check:")
    for name in dense_params:
        if "moe" not in name and name in moe_params:
            # 转换为float32确保比较精度
            param_equal = torch.allclose(dense_params[name].float(), 
                                       moe_params[name].float(), 
                                       atol=1e-5)
            status = "MATCH" if param_equal else "MISMATCH"
            print(f"{name:60} {status}")
            
def check_expert_parameters(dense_model, moe_model):
    """检查专家层参数复制情况"""
    dense_params = dict(dense_model.named_parameters())
    moe_params = dict(moe_model.named_parameters())
    
    print("="*40)
    print("Expert Parameters Check:")
    
    # 检查专家复制
    expert_layers = [k for k in moe_params if "moe.experts" in k]
    
    if not expert_layers:
        print("No MoE experts found in the model. Please check the model structure.")
        return
    
    for expert_param in expert_layers:
        # 获取对应的原始参数名（experts.0）
        original_name = expert_param.replace(".experts.1", ".experts.0")\
                                    .replace(".experts.2", ".experts.0")\
                                    .replace(".experts.3", ".experts.0")
        
        # 检查是否复制自原始参数
        expert_idx = int(expert_param.split(".")[5])
        if expert_idx == 0:
            # 第一个专家应该完全一致
            param_equal = torch.allclose(dense_params[original_name].float(),
                                       moe_params[expert_param].float(),
                                       atol=1e-5)
            status = "MATCH(orig)" if param_equal else "MISMATCH(orig)"
        else:
            # 其他专家应该相似但有差异
            similarity = torch.mean(torch.abs(
                dense_params[original_name].float() - 
                moe_params[expert_param].float()
            ))
            status = f"SIMILAR({similarity:.6f})" if similarity < 0.1 else "NOISE_TOO_LARGE"
            
        print(f"{expert_param:60} {status}")
        
    # 检查门控参数初始化
    print("-"*50)
    
    # 查找 router 参数
    router_param = None
    for name in moe_params:
        if "router.weight" in name:
            router_param = moe_params[name]
            break
    
    if router_param is not None:
        router_weight = router_param.float()
        print("Router weight stats:")
        print(f"  Mean: {router_weight.mean().item():.6f}")
        print(f"  Std: {router_weight.std().item():.6f}")
    else:
        print("Router weight not found. Please check the model structure.")
    
    # 检查gate_proj初始化
    for name in moe_params:
        if "gate_proj.weight" in name:
            gate = moe_params[name].float()
            fan_in = gate.size(1)
            expected_std = torch.sqrt(torch.tensor(2. / fan_in))
            actual_std = gate.std()
            print(f"Gate projection {name}:")
            print(f"  Expected std (~{expected_std:.4f}): Actual std {actual_std:.4f}")

# 执行检查
check_common_parameters(dense_model, moe_model)
check_expert_parameters(dense_model, moe_model)