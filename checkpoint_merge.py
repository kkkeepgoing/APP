import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

def apply_app_algorithm(pretrained_weights, finetuned_weights, rank_k=64):

    app_state_dict = {}
    
    # 遍历每一层权重 (例如: weight, bias)
    for key in pretrained_weights.keys():
        W_pre = pretrained_weights[key].float() # 原始权重
        W_ft = finetuned_weights[key].float()   # 微调后权重
        
        # 1. 计算任务向量 (Task Vector)
        tau = W_ft - W_pre
        
        # 2. 对任务向量进行 SVD 分解
        # 注意：对于 1D 向量 (如 bias)，SVD 没有意义，直接保留或不做处理
        if len(tau.shape) > 1: 
            U, S, Vh = torch.linalg.svd(tau, full_matrices=False)
            
            # 3. 基于秩的截断 (Rank-based Truncation)
            # 只保留前 k 个最大的奇异值
            # 论文中 k 的取值通常较小，如 32, 64, 128，取决于具体层的大小
            actual_k = min(rank_k, S.shape[0])
            
            U_k = U[:, :actual_k]
            S_k = torch.diag(S[:actual_k])
            Vh_k = Vh[:actual_k, :]
            
            # 重构截断后的任务向量
            tau_app = U_k @ S_k @ Vh_k
        else:
            # 对于 bias 向量，通常可以直接保留微调结果，或者也做缩放
            # 论文主要针对权重矩阵，这里简单起见保留微调后的 bias
            tau_app = tau 
            
        # 4. 将过滤后的更新加回原始权重
        W_app = W_pre + tau_app
        
        # 转回原来的数据类型 (如 bfloat16)
        app_state_dict[key] = W_app.to(pretrained_weights[key].dtype)
        
    return app_state_dict

# --- 使用示例 ---
# 1. 获取微调后的权重
model1 = Qwen3VLForConditionalGeneration.from_pretrained("finetuned_model_path", device_map="auto", torch_dtype=torch.bfloat16)
# finetuned_projector_weights = model1.get_model().mm_projector.state_dict()
finetuned_projector_weights = model1.model.visual.merger.state_dict()

model2 = Qwen3VLForConditionalGeneration.from_pretrained("base_model_path", device_map="auto", torch_dtype=torch.bfloat16)
# pretrained_projector_weights = model2.get_model().mm_projector.state_dict()
pretrained_projector_weights = model2.model.visual.merger.state_dict()

# 2. 执行 APP 算法
# rank_k 是超参数，论文建议通过验证集搜索，通常 64 或 128 是不错的起点
final_weights = apply_app_algorithm(
    pretrained_projector_weights, 
    finetuned_projector_weights, 
    rank_k=64
)

# 3. 将处理后的权重加载回模型并保存
model1.model.visual.merger.load_state_dict(final_weights)
model1.save_pretrained("new_model_path")
