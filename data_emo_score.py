# coding=utf-8
"""读取情感得分，将消极得分2改为-1"""

import pandas as pd
import os

def main():
    # 文件路径
    data_dir = 'peizhen_data'
    input_csv = os.path.join(data_dir, 'data_topic_emo.csv')
    output_csv = os.path.join(data_dir, 'result.csv')
    
    print(f"读取文件: {input_csv}")
    
    # 读取 CSV 文件
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding='gbk')
    
    print(f"共读取 {len(df)} 条数据")
    
    # 检查是否存在"情感得分"列
    if '情感得分' not in df.columns:
        print("错误: 找不到'情感得分'列")
        return
    
    # 查看原始情感分布
    print("\n原始情感分布:")
    print(df['情感得分'].value_counts().sort_index())
    
    # 将情感得分2改为-1
    df['情感得分'] = df['情感得分'].replace(2, -1)
    
    # 查看修改后的情感分布
    print("\n修改后情感分布:")
    print(df['情感得分'].value_counts().sort_index())
    
    # 保存结果
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_csv}")
    
    print("\n处理完成!")

if __name__ == "__main__":
    main()
