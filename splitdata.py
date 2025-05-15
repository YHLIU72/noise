import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 Excel 文件中的所有工作表
file_path = '../data/data_final_fft_0318.xlsx'  
sheets_dict = pd.read_excel(file_path, sheet_name=None)  # 读取所有工作表

# 创建两个列表，用于存储所有工作表的训练集和测试集
all_train_dfs = []
all_test_dfs = []

# 遍历每个工作表
for sheet_name, df in sheets_dict.items():

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['type'], random_state=42)

    # 将划分结果添加到列表中
    all_train_dfs.append(train_df)
    all_test_dfs.append(test_df)

# 将每个工作表的训练集和测试集分别保存到不同的工作表中
train_output_file = 'train0318.xlsx'
test_output_file = 'test0318.xlsx'

with pd.ExcelWriter(train_output_file) as writer:
    for sheet_name, train_df in zip(sheets_dict.keys(), all_train_dfs):
        train_df.to_excel(writer, sheet_name=sheet_name, index=False)

with pd.ExcelWriter(test_output_file) as writer:
    for sheet_name, test_df in zip(sheets_dict.keys(), all_test_dfs):
        test_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"每个工作表的划分结果已分别保存到文件: {test_output_file} 和 {train_output_file}")