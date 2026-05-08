"""
Step 2: 生成 LLM 电影语义向量
功能：调用智谱 Embedding-3 API，把电影文本转成1024维向量，存成 npy 文件
这个文件只需要跑一次，后面所有训练直接加载 npy 文件
"""

import pandas as pd
import numpy as np
import time
from zhipuai import ZhipuAI

# ── 0. 配置 ────────────────────────────────────────────────────────────────
API_KEY   = "你的智谱API_KEY"        # ← 替换成你的 key
DATA_DIR  = "./data/processed"
OUT_DIR   = "./embeddings"
DIMENSION = 1024                      # 选择1024维，原因见文档分析
BATCH_SIZE = 64                       # 智谱单次最多64条

import os
os.makedirs(OUT_DIR, exist_ok=True)

client = ZhipuAI(api_key=API_KEY)

# ── 1. 读取电影文本映射表 ──────────────────────────────────────────────────
movie_text_map = pd.read_csv(f"{DATA_DIR}/movie_text_map.csv", index_col="movie_id_encoded")
print(f"共 {len(movie_text_map)} 部电影需要生成向量")
print(movie_text_map.head(3))

# ── 2. 批量调用 API ────────────────────────────────────────────────────────
# 为什么要批量？
# 单条循环调用：3883次 HTTP 请求，慢且容易触发频率限制
# 批量调用：约61次请求，快10-20倍

movie_ids   = movie_text_map.index.tolist()
texts       = movie_text_map["llm_input_text"].tolist()
embeddings  = {}   # {movie_id_encoded: numpy_array}

total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(texts), BATCH_SIZE):
    batch_ids   = movie_ids[i : i + BATCH_SIZE]
    batch_texts = texts[i : i + BATCH_SIZE]
    batch_num   = i // BATCH_SIZE + 1

    try:
        response = client.embeddings.create(
            model      = "embedding-3",
            input      = batch_texts,
            dimensions = DIMENSION,    # 指定1024维输出
        )

        # response.data 是一个列表，顺序和输入对应
        for j, emb_obj in enumerate(response.data):
            embeddings[batch_ids[j]] = np.array(emb_obj.embedding, dtype=np.float32)

        print(f"批次 {batch_num}/{total_batches} 完成，已处理 {min(i+BATCH_SIZE, len(texts))} 部电影")

    except Exception as e:
        print(f"批次 {batch_num} 失败: {e}")
        # 失败了等2秒再继续，不中断整个流程
        time.sleep(2)
        continue

    # 每批之间稍微停一下，避免触发频率限制
    time.sleep(0.3)

print(f"\n成功生成 {len(embeddings)} / {len(movie_ids)} 部电影的向量")

# ── 3. 检查是否有遗漏 ──────────────────────────────────────────────────────
missing = set(movie_ids) - set(embeddings.keys())
if missing:
    print(f"⚠️  有 {len(missing)} 部电影未生成向量，movie_id: {list(missing)[:10]}")
else:
    print("✅ 所有电影向量生成完毕，无遗漏")

# ── 4. 整理成矩阵并保存 ───────────────────────────────────────────────────
# 为什么存成矩阵而不是字典？
# 训练时需要根据 movie_id 快速查询向量
# 矩阵的行索引 = movie_id_encoded，直接用 matrix[movie_id] 取向量，O(1) 速度

max_id     = max(embeddings.keys())
emb_matrix = np.zeros((max_id + 1, DIMENSION), dtype=np.float32)

for mid, vec in embeddings.items():
    emb_matrix[mid] = vec

np.save(f"{OUT_DIR}/movie_embeddings.npy", emb_matrix)
print(f"\n向量矩阵已保存：shape = {emb_matrix.shape}")
print(f"路径：{OUT_DIR}/movie_embeddings.npy")

# ── 5. 简单验证：看两部电影的语义是否合理 ────────────────────────────────
# 用余弦相似度检验：同类型电影应该比不同类型电影更相似

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 取前3部电影互相比较
ids_to_check = movie_ids[:3]
print("\n── 语义验证（余弦相似度）──")
for i in range(len(ids_to_check)):
    for j in range(i+1, len(ids_to_check)):
        id_i, id_j = ids_to_check[i], ids_to_check[j]
        sim = cosine_sim(embeddings[id_i], embeddings[id_j])
        text_i = movie_text_map.loc[id_i, "llm_input_text"][:40]
        text_j = movie_text_map.loc[id_j, "llm_input_text"][:40]
        print(f"  {text_i}")
        print(f"  {text_j}")
        print(f"  相似度: {sim:.4f}\n")