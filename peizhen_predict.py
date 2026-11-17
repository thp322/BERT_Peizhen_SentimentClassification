# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import predict
import tensorflow as tf

# 禁用eager execution
tf.compat.v1.disable_eager_execution()

def main():
    # 设置参数
    FLAGS = predict.FLAGS
    
    # 必须设置的参数
    FLAGS.data_dir = 'peizhen_data'
    FLAGS.task_name = 'sim'
    FLAGS.vocab_file = './GLUE/BERT_BASE_DIR/chinese_L-12_H-768_A-12/vocab.txt'
    FLAGS.bert_config_file = './GLUE/BERT_BASE_DIR/chinese_L-12_H-768_A-12/bert_config.json'
    FLAGS.init_checkpoint = 'sim_model'
    FLAGS.output_dir = 'peizhen_output'
    FLAGS.do_train = False
    FLAGS.do_eval = False
    FLAGS.do_predict = True
    FLAGS.max_seq_length = 70
    FLAGS.predict_batch_size = 32
    FLAGS.do_lower_case = True
    FLAGS.use_tpu = False
    
    # 读取数据
    input_file = os.path.join(FLAGS.data_dir, 'data_topic.csv')
    print(f"正在读取文件: {input_file}")
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')
    
    print(f"共读取 {len(df)} 条数据")
    
    # 获取内容列
    if '内容' not in df.columns:
        print("错误: 找不到'内容'列")
        return
    
    texts = df['内容'].astype(str).tolist()
    
    # 进行预测
    print("开始进行情感预测...")
    results = predict.predicts(texts)
    
    # 解析结果
    # results 格式: {text: [label, probability], ...}
    emotion_labels = []
    emotion_probs = []
    
    for text in texts:
        if text in results:
            label, prob = results[text]
            emotion_labels.append(label)
            emotion_probs.append(prob)
        else:
            emotion_labels.append(None)
            emotion_probs.append(None)
    
    df['情感得分'] = emotion_labels
    df['情感概率'] = emotion_probs
    
    # 保存结果
    output_file = os.path.join(FLAGS.data_dir, 'data_topic_emo.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到: {output_file}")
    
    # 统计信息
    print("\n情感分布统计:")
    print(df['情感得分'].value_counts())
    
    print("\n预测完成!")

if __name__ == "__main__":
    main()
