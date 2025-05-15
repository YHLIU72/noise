import torch

# 模拟数据
# 假设 batch_size 为 3，preds 张量的形状为 (3, 4, 5, 1)
preds = torch.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
print("原始 preds 张量:")
print(preds)
print("原始 preds 形状:", preds.shape)

# 模拟 sheet_idx 张量
sheet_idx = torch.tensor([1, 2, 3]).unsqueeze(1)
print("\nsheet_idx 张量:")
print(sheet_idx)

# 模拟 types 张量
types = torch.tensor([2, 1, 0])
print("\ntypes 张量:")
print(types)

# 生成 batch_indices
batch_indices = torch.arange(preds.size(0))

# 第一步：根据 batch_indices 和 sheet_idx 选择元素
preds = preds[batch_indices, sheet_idx.squeeze(), types, :]
print("\n第一步操作后的 preds 张量:")
print(preds)
print("第一步操作后 preds 形状:", preds.shape)



print("\n最终的 preds 张量:")
print(preds)
print("最终 preds 形状:", preds.shape)