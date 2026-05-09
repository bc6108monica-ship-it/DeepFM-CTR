"""
Step 1: 数据预处理
功能：读取 ml-1m 原始数据，合并、清洗、编码、切分，输出 train/test csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# ── 0. 路径配置，改成你自己的路径 ─────────────────────────────────────────
RAW_DIR = "./data/raw"          # ← 改成你本地 ml-1m 文件夹路径
OUT_DIR = "./data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. 读取三个 dat 文件 ───────────────────────────────────────────────────
# MovieLens 1M 用 `::` 分隔，pandas 不直接支持多字符分隔符，用 engine='python'

ratings = pd.read_csv(
    f"{RAW_DIR}/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

users = pd.read_csv(
    f"{RAW_DIR}/users.dat",
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zip"],
)

movies = pd.read_csv(
    f"{RAW_DIR}/movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1",   # 部分电影标题含特殊字符
)

print(f"ratings: {ratings.shape}")   # 应为 (1000209, 4)
print(f"users:   {users.shape}")     # 应为 (6040, 5)
print(f"movies:  {movies.shape}")    # 应为 (3883, 3)

# ── 2. 合并成一张大表 ──────────────────────────────────────────────────────
#dat数据读进来之后变成了 DataFrame了，刷SQL的时候pandas没白刷
df = ratings.merge(users, on="user_id").merge(movies, on="movie_id")
print(f"\n合并后: {df.shape}")

# ── 3. 构造二分类 label ────────────────────────────────────────────────────
# rating >= 4 视为正样本（喜欢），否则为负样本
# 业务理解：4星以上才算真正的点击/转化行为
#里面是bool类型的条件判断，结果是一个布尔 Series，astype(int) 将 True 转为 1，False 转为 0，
# 最终得到一个二分类的 label 列，就是这个新定义的列label，1表示用户喜欢这部电影（评分4或5），0表示不喜欢（评分1、2、3）
df["label"] = (df["rating"] >= 4).astype(int)
print(f"\nLabel 分布:\n{df['label'].value_counts()}")

# ── 4. 给每部电影打上冷启动标签 ───────────────────────────────────────────
# 用全量数据统计每部电影的交互次数（注意：实际工程中应只用训练集统计，这里简化处理）
#这里直接用原始数据，相当于是训练集和测试集混在一起了，实际工程中应该先切分再统计，这里为了简化流程直接统计全量数据，后续评估时再分析不同冷启动层级的表现
movie_count = df.groupby("movie_id")["rating"].count().rename("movie_interaction_count")
#这个movie_count是一个Series，索引是movie_id，值是该电影的交互次数。通过merge把这个统计结果合并回原始的df中，得到每行数据对应的电影交互次数，
#看起来像dataframe，但是其实是series，merge后就变成了dataframe了，movie_id是连接键，movie_interaction_count是新添加的列，表示每部电影的交互次数
df = df.merge(movie_count, on="movie_id")

# 冷启动分层：后面第5步评估会用到
# < 5     → 极冷 (very_cold)
# 5 ~ 20  → 冷   (cold)
# >= 20   → 热   (warm)
#定义了一个名为 coldstart_label 的函数，它接受一个参数 n。
def coldstart_label(n):
    if n < 5:
        return "very_cold"
    elif n < 20:
        return "cold"
    else:
        return "warm"

#这行是在电影交互次数的基础上打冷启动标签，后续评估会用到这个标签来分析模型在不同冷启动层级的表现
#使用 apply 方法将 coldstart_label 函数应用到 df 的 movie_interaction_count 列的每个元素上，
# 根据每个电影的交互次数生成对应的冷启动标签，并将结果赋值给 df 的新列 coldstart_tier。
df["coldstart_tier"] = df["movie_interaction_count"].apply(coldstart_label)
print(f"\n冷启动分层分布:\n{df['coldstart_tier'].value_counts()}")

# ── 5. 特征工程：Label Encoding ────────────────────────────────────────────
# DeepCTR-Torch 的 SparseFeat 要求特征值是从 0 开始的连续整数
# 原始的 user_id / movie_id 不连续，必须重新编码

#定义了一个新的列表 sparse_features，包含了所有的稀疏特征（类别型特征）的列名。
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip"]

lbe_dict = {}  # 存所有特征的encoder，以防万一
for feat in sparse_features:
    lbe = LabelEncoder()
    #df[feat] = — 用编码后的新值覆盖原来的列。
    df[feat] = lbe.fit_transform(df[feat].astype(str))
    #fit_transform 是 LabelEncoder 的一个方法，它会先拟合（fit）数据，找到所有唯一的类别并为它们分配一个整数标签，然后对数据进行转换（transform），
    # 将原始的类别值替换为对应的整数标签。 
    #这些列全都给编码了，astype(str) 是为了统一类型，LabelEncoder 对混合类型容易报错
    # astype(str) 是为了统一类型，LabelEncoder 对混合类型容易报错
    lbe_dict[feat] = lbe  # 每个都存下来，以防后续需要反编码或者对新数据进行同样的编码

# 验证编码结果
for feat in sparse_features:
    print(f"{feat}: {df[feat].nunique()} unique values, range [{df[feat].min()}, {df[feat].max()}]")

"""
全都得改因为这个encoder没对齐，但是可以留着看语法
# ── 6. 为 LLM 向量生成准备：保存 movie_id → 原始文本 的映射 ──────────────
# 注意：这里用编码前的原始 movie_id 对应文本，所以要在编码前做映射
# 重新读一次 movies 获取原始信息（编码前的 movie_id）
movies_raw = pd.read_csv(
    f"{RAW_DIR}/movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    #encoding指定编码格式，ml-1m数据集中电影标题包含一些特殊字符，使用默认的utf-8编码可能会导致读取错误，因此指定为latin-1编码可以正确读取这些特殊字符。
    #编码格式决定了计算机如何将存储的二进制数据解释为字符，一共有三种常见的编码格式：ASCII、UTF-8 和 Latin-1（ISO-8859-1）。
    # ASCII 是最基本的编码，只能表示 128 个字符，适用于英文文本；
    # UTF-8 是一种变长编码，可以表示全球范围内的所有字符，适用于多语言文本；
    # Latin-1 是一种单字节编码，可以表示西欧语言中的字符，适用于包含特殊字符的文本。
    #这里选择Latin是因为 movies.dat 文件是使用 Latin - 1 编码保存的
    encoding="latin-1",
)

# 构造 LLM 输入文本：标题 + 类型
# 例："Toy Story (1995) | Animation Children's Comedy"
#后面的replace是把genres列中的|替换成空格，这样就不会干扰LLM的理解了，毕竟|在文本中可能没有明确的语义，反而会增加理解难度
movies_raw["llm_input_text"] = movies_raw["title"] + " | " + movies_raw["genres"].str.replace("|", " ")

# 同时保存编码后的 movie_id（用于后面对齐向量）
lbe_movie = LabelEncoder()
movies_raw["movie_id_encoded"] = lbe_movie.fit_transform(movies_raw["movie_id"].astype(str))

#set_index 是 pandas 中的一个方法，用于将 DataFrame 的某一列设置为索引。
#这里的 movie_id_encoded 是经过 LabelEncoder 编码后的电影 ID 列，llm_input_text 是构造的 LLM 输入文本列。
#movie_text_map 就变成了一个以 movie_id_encoded 为索引，llm_input_text 为数据列的 DataFrame。
movie_text_map = movies_raw[["movie_id_encoded", "llm_input_text"]].set_index("movie_id_encoded")
#将现在的这个映射表保存为csv文件到指定的这个路径，
movie_text_map.to_csv(f"{OUT_DIR}/movie_text_map.csv")
print(f"\n电影文本映射表已保存，共 {len(movie_text_map)} 部电影")
print(movie_text_map.head(3))
"""

# ── 6. 为 LLM 向量生成准备：保存 movie_id → 原始文本 的映射 ──────────────
#lbe_movie = LabelEncoder()
#df["movie_id"] = lbe_movie.fit_transform(df["movie_id"].astype(str))
# 存 lbe_movie 供第二步使用
#joblib.dump(lbe_movie, f"{OUT_DIR}/lbe_movie.pkl")

#我真服了这个变量名字全都不对我还得改
# 存电影文本映射，用的是 merge 后已编码的 movie_id
movie_text_map = df[["movie_id", "title", "genres"]].drop_duplicates("movie_id").copy()
movie_text_map = movie_text_map.rename(columns = {'movie_id': 'movie_id_encoded'})
movie_text_map["llm_input_text"] = movie_text_map["title"] + " | " + movie_text_map["genres"].str.replace("|", " ")
movie_text_map[["movie_id_encoded", "llm_input_text"]].to_csv(f"{OUT_DIR}/movie_text_map.csv", index=False)
print(f"\n电影文本映射表已保存，共 {len(movie_text_map)} 部电影")
print(movie_text_map.head(3))




# ── 7. 按时间戳排序后切分 train / test ────────────────────────────────────
# 为什么不用随机切分？
# 因为推荐系统是时序问题：训练集必须是"过去"，测试集必须是"未来"
# 随机切分会造成数据泄漏（用未来数据预测过去）
#这回直接就是在这个合并过的大表df上切了
df = df.sort_values("timestamp").reset_index(drop=True)

split_idx = int(len(df) * 0.8)   # 80% 训练，20% 测试
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

print(f"\ntrain: {train.shape}, test: {test.shape}")
#因为 label 是二分类的（0 或 1），所以它的均值其实就是正样本占比：
#训练集正样本占比
print(f"train label 均值: {train['label'].mean():.4f}")
#测试集正样本占比
print(f"test  label 均值: {test['label'].mean():.4f}")

# ── 8. 保存 ───────────────────────────────────────────────────────────────
train.to_csv(f"{OUT_DIR}/train.csv", index=False)
test.to_csv(f"{OUT_DIR}/test.csv",  index=False)
print(f"\n✅ 预处理完成，文件保存至 {OUT_DIR}/")