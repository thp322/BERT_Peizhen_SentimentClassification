# coding=utf-8

import pandas as pd
import os

def main():
    data_dir = 'peizhen_data'
    output_dir = 'peizhen_output'
    
    input_csv = os.path.join(data_dir, 'data_topic.csv')
    result_tsv = os.path.join(output_dir, 'test_results.tsv')
    output_csv = os.path.join(data_dir, 'data_topic_emo.csv')
    
    print(f"读取原始数据: {input_csv}")
    
    # 原始 CSV 文件
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding='gbk')
    
    print(f"共读取 {len(df)} 条数据")
    
    # 预测结果 TSV 文件
    print(f"读取预测结果: {result_tsv}")
    
    # TSV 文件格式: 每行是三个概率值（0, 1, 2）
    probabilities = []
    with open(result_tsv, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                probs = [float(x) for x in line.split('\t')]
                probabilities.append(probs)
    
    print(f"共读取 {len(probabilities)} 条预测结果")
    
    # 检查数据数量是否匹配
    if len(df) != len(probabilities):
        print(f"警告: 数据数量不匹配! CSV: {len(df)}, TSV: {len(probabilities)}")
        # 使用较小的数量
        min_len = min(len(df), len(probabilities))
        df = df.iloc[:min_len]
        probabilities = probabilities[:min_len]
    
    # 解析情感得分
    emotion_labels = []
    emotion_probs = []
    
    label_list = ['0', '1', '2']  # 情感类别
    
    for probs in probabilities:
        # 找到概率最大的类别
        max_idx = probs.index(max(probs))
        label = label_list[max_idx]
        prob = probs[max_idx]
        
        emotion_labels.append(label)
        emotion_probs.append(prob)
    
    df['情感得分'] = emotion_labels
    df['情感概率'] = emotion_probs
    
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_csv}")
    
    # 统计信息
    print("\n情感分布统计:")
    print(df['情感得分'].value_counts().sort_index())
    
    print("\n处理完成!")

if __name__ == "__main__":
    main()
