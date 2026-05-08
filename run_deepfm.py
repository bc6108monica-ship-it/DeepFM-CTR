import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 导入 DeepCTR-Torch 核心组件
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM

if __name__ == "__main__":
    # 1. 加载数据
    data = pd.read_csv('./criteo_sample.txt')
    
    # 划分稀疏特征（类别型）和稠密特征（数值型）
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    # 填补缺失值
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)
    target = ['label']

    # 2. 数据预处理
    # 类别特征进行 Label编码 (0, 1, 2, ...)
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        
    # 数值特征进行 0-1 归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 3. 告诉模型哪些是类别特征，哪些是数值特征
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + \
                             [DenseFeat(feat, 1,)
                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 4. 划分训练集和测试集
    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 5. 定义 DeepFM 模型（支持 GPU/CPU 自动切换）
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)
    
    # 编译模型（配置优化器、损失函数、评估指标）
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', "auc"])

    # 6. 开始训练！
    print("开始训练...")
    model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, validation_split=0.2, verbose=2)

    # 7. 预测与评估
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("测试集 LogLoss:", round(log_loss(test[target].values, pred_ans), 4))
    print("测试集 AUC:", round(roc_auc_score(test[target].values, pred_ans), 4))