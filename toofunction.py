import pandas as pd

def calculate_mean_and_std_all_sheets(file_path, column_name):
    """
    Calculate the mean and standard deviation of a specific column across all sheets in an Excel file.

    :param file_path: Path to the Excel file
    :param column_name: Name of the column to calculate statistics for
    :return: A tuple containing the mean and standard deviation
    """
    try:
        # Read all sheets from the Excel file
        all_sheets = pd.read_excel(file_path, sheet_name=None)

        # Combine data from all sheets
        combined_data = pd.DataFrame()
        for sheet_name, sheet_data in all_sheets.items():
            # 修复：将列数据转换为 DataFrame 后再合并
            combined_data = pd.concat([combined_data, pd.DataFrame({column_name: sheet_data[column_name]})], ignore_index=True)

        # Check if the specified column exists in the combined data
        if column_name not in combined_data.columns:
            print(f"Error: Column '{column_name}' not found in the Excel file.")
            return None, None

        # Calculate mean and standard deviation
        mean_value = combined_data[column_name].mean()
        std_value = combined_data[column_name].std()

        return mean_value, std_value
    except Exception as e:
        print(f"Error: {e}")
        return None, None
if __name__ == "__main__":
    file_path = "../data/test0318.xlsx"  # Replace with your Excel file path
    column_names=["Qv","DP","RPM","wheel","height","hole"]
    for column_name in column_names:
         # Replace with the column name you want to calculate statistics for
        mean_value, std_value = calculate_mean_and_std_all_sheets(file_path, column_name)
        if mean_value is not None and std_value is not None:
            print(f"{column_name}: Mean: {mean_value:.2f}, Standard Deviation: {std_value:.2f}")
# Example usage:
# file_path = "example.xlsx"
# column_name = "Column1"
# mean, std = calculate_mean_and_std_all_sheets(file_path, column_name)
# print(f"Mean: {mean}, Standard Deviation: {std}")