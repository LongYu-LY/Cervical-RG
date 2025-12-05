import shutil
import argparse
import torch
from Jamba.configuration_jamba import JambaConfig
from Jamba.modeling_jamba import JambaForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the input model")
parser.add_argument("--output_path", type=str, help="Path to save the output model")
args = parser.parse_args()

# 加载原始模型和配置
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
original_config = JambaConfig.from_pretrained(args.model_path, trust_remote_code=True).to_dict()

# 修改配置为MoE模型
moe_config = original_config.copy()
# moe_config.update({
#     "num_experts": 4,          # 专家数量
#     "num_experts_per_tok": 2,  # 每个token使用的专家数
# })
moe_config["num_experts"] = 4
moe_config["num_experts_per_tok"] = 2
def name_mapping(name):
    """
    新的参数映射逻辑：
    1. 对于非专家层参数，直接复制
    2. 对于专家层参数，复制4份专家参数
    """
    if "moe.experts" in name:
        # 处理专家层参数
        base_name = name.replace("moe.experts.0", "moe.experts.{}")
        return [base_name.format(i) for i in range(4)], False

    return [name], False

# 加载原始dense模型
dense_model = JambaForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
param_dict = dict(dense_model.named_parameters())
# print(param_dict)
# 创建新的MoE模型
moe_model = JambaForCausalLM(JambaConfig(**moe_config))
param_dict_moe = dict(moe_model.named_parameters())
# print(moe_model.name())

# 参数迁移
# 修改后的参数复制逻辑
for name, param in moe_model.named_parameters():
    src_names, is_gate = name_mapping(name)
    
    # if is_gate:
    #     # 门控层直接复制原始参数
    #     param.data.copy_(param_dict[src_names[0]].data)
    if "moe.experts" in name:
        # 关键修改：所有专家都复制原始experts.0的参数
        if "gate_proj.weight" in name:
            torch.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='linear') 
        expert_index = int(name.split(".")[5])
        original_name = name.replace(f".experts.{expert_index}", ".experts.0")  # 保持原始参数路径
          # 获取新模型的专家编号
        if expert_index == 0:
            # 第一个专家直接复制
            param.data.copy_(param_dict[original_name].data)
        else:
            # 其他专家复制原始参数（可添加随机初始化差异）
            param.data.copy_(param_dict[original_name].data.clone())
            # 可选：添加微小差异防止完全相同的专家
            param.data += torch.randn_like(param.data) * 0.01
    elif "moe.router.weight" in name:
        param.data = torch.nn.init.xavier_uniform_(param.data)
    else:
        if name not in param_dict:
            raise KeyError(f"Missing parameter: {name}, 需要检查：\n"
                        "1. 是否错误引用了MoE特有参数\n"
                        "2. 原始模型是否包含对应层\n"
                        "3. 参数命名是否匹配（建议打印param_dict.keys()对比）")
        param.data.copy_(param_dict[name].data)

# 保存配置和模型
moe_model.to(dtype=torch.bfloat16)
tokenizer.save_pretrained(args.output_path)
moe_model.save_pretrained(args.output_path)

# 复制必要文件
shutil.copy("Jamba/configuration_jamba.py", args.output_path)
shutil.copy("Jamba/modeling_jamba.py", args.output_path)