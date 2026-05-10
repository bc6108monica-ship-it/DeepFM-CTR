# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DeepFM 是什么
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2017 年华为提出的 CTR 预测模型，核心创新：
用 FM 代替 Wide & Deep 的 Wide 侧，让模型自动学二阶交叉，省掉人工特征工程。

模型分三部分，前向传播时相加：
  logit = Linear(原始输入) + FM(Embedding 两两内积) + DNN(Embedding 拼接 → 全连接)
  y_pred = sigmoid(logit)

                      ┌─────────────────────┐
  输入 ──→ Linear ──→ │          +           │ ──→ sigmoid ──→ 输出
            │         │                      │
            ├─→ Embedding ──→ FM ───────────→│
            │              ──→ DNN ─────────→│
                      └─────────────────────┘

  ① Linear  — 每个特征自己的权重，学"看过这部电影"这类单特征信号
  ② FM      — 特征两两内积，学"男性 + 科幻"这类二阶组合
  ③ DNN     — Embedding 拼接后过全连接，学"男性 + 科幻 + 18-25岁"这类高阶交互

优势：相比 Wide & Deep 不需要手动构造交叉特征，FM 自动搞定。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import FM, DNN


class DeepFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        # ── BaseModel 初始化 ────────────────────────────────────────────────
        # 基类 BaseModel 做三件事：
        #   1. 根据 feature_columns 构建 Embedding 字典（每个类别特征一张 embedding 表）
        #   2. 构建线性部分 Linear（SparseFeat 查 embedding 再求和，DenseFeat 直接加）
        #   3. 设置优化器、损失函数、评估指标（在 compile() 时调用）
        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0

        # ── ② FM 部分 ──────────────────────────────────────────────────────
        # FM 层做的计算：对每对特征 embedding 做内积再求和
        #   公式: Σ_i Σ_{j>i} <v_i, v_j> * x_i * x_j
        #   其中 v_i 是特征 i 的 embedding 向量，x_i 是特征 i 的值（类别特征为0/1）
        # 代码见 deepctr_torch/layers/fm.py
        if use_fm:
            self.fm = FM()

        # ── ③ DNN 部分 ─────────────────────────────────────────────────────
        # DNN 层做的计算：
        #   1. 把所有特征的 embedding 拼接成一个长向量
        #   2. 过 N 层全连接：256 → 128 → ...
        #   3. 最后一层接一个 Linear(128 → 1) 输出标量 logit
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            # 把 DNN 的权重加入 L2 正则化跟踪列表
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        # ── 前向传播：三路相加 ──────────────────────────────────────────────

        # 1. 把所有特征转成 embedding（类别查表，数值直接拿）
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        # ① Linear 部分：每个特征的 embedding 直接求和 + Dense 特征加权
        #    等价于 Logistic Regression：W * X + b
        logit = self.linear_model(X)

        # ② FM 部分：embedding 两两内积，捕获二阶交叉信号
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        # ③ DNN 部分：拼接所有 embedding → 全连接 → 标量
        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        # sigmoid 输出 0~1 概率
        y_pred = self.out(logit)

        return y_pred
