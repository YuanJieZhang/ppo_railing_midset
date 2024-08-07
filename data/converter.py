import pandas as pd

# 读取CSV文件
df = pd.read_csv('bay_vio_data_03_19.csv')

# 更新 'aim_maker' 列
def update_aim_maker(row):
    # 提取 street_marker 列的数字部分
    street_number = int(row['street_marker'][1:])
    aim_num = int(row['aim_marker'][1:])
    if aim_num < 5000:
        print(aim_num)
    # 计算 aim_maker 新的值
    new_aim_number = street_number + 5000
    # 返回新的 aim_maker 值
    return 'A' + str(new_aim_number)

# 应用函数到每一行
df['aim_marker'] = df.apply(update_aim_maker, axis=1)

# 将修改后的DataFrame保存回CSV文件
df.to_csv('bay_vio_data_03_19.csv', index=False)