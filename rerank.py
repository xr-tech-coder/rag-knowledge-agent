"""实验 5：重排（cross-encoder）。

要回答的问题：混合检索的召回已经 78/78 全中，但 top1 还有 2 道错（22、24）。
    正确片段在 top2，模型靠自己往后翻一页答对的。换个弱模型、或者 prompt 少一句话，
    就可能翻错。重排能不能把 top1 修对？

为什么这里只能用 cross-encoder，不能用向量：
    bi-encoder 把问句和片段分别编码成两个向量再算余弦，两者从头到尾没见过面；
    cross-encoder 把 (问句, 片段) 拼成一个序列过一遍模型，能看到词与词之间的交互。
    代价是每个候选都要单独算一次，10 个候选 = 10 次前向。慢，但准。

跑法：
    .venv\\Scripts\\python.exe rerank.py

依赖：
    sentence-transformers + BAAI/bge-reranker-base（约 1.1GB，首次运行会下载）
    想换模型：set RERANK_MODEL=BAAI/bge-reranker-v2-m3
"""
import json
import os
import time

import numpy as np

import hybrid
import retrieval as R

CAND_K = 10                                    # 候选池大小：先粗排 10 个，再精排
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
HF_HOME = os.path.join(R.HERE, ".models")      # 权重落在项目盘，不占 C 盘

# 本地权重候选：(目录, 该文件至少多大才算下完整)。
# bge-reranker-base 的 1.06GB 在国内镜像反复 502，只下到 404MB；
# mmarco 是同样支持中文的轻量替代（117M 参数），所以把它排在前面。
LOCAL_WEIGHTS = [
    (".models/mmarco-reranker", 470_000_000),
    (".models/bge-reranker-base", 1_100_000_000),
]


def load_reranker():
    """加载 cross-encoder。

    先找本地完整副本，找不到再回落到 HuggingFace 缓存。
    必须校验文件大小 —— 半截的权重也能通过「config.json 存在」这种检查，
    加载时才炸，报错信息还和下载无关。
    """
    from sentence_transformers import CrossEncoder

    for rel, min_size in LOCAL_WEIGHTS:
        d = os.path.join(R.HERE, rel)
        w = os.path.join(d, "model.safetensors")
        if os.path.exists(os.path.join(d, "config.json")) and os.path.exists(w) \
                and os.path.getsize(w) >= min_size:
            print(f"  使用本地权重 {rel}（{os.path.getsize(w) / 1e6:.0f}MB）")
            return CrossEncoder(d), rel

    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return CrossEncoder(RERANK_MODEL), RERANK_MODEL


def rerank(model, question, cands, top_n=R.TOP_K):
    """给每个候选打分并重排。

    model.predict 吃的是一批 (问句, 片段) 对，返回同长度的相关性分数。
    分数不是概率，只有相对大小有意义 —— 这点和 BM25 / RRF 都不一样，
    所以跨模式的分数不能放一起比。
    """
    if not cands:
        return []
    pairs = [[question, t] for _, _, t in cands]
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores))[:top_n]
    return [(cands[i][0], float(scores[i]), cands[i][2]) for i in order]


def main():
    chunks = R.load_chunks()
    bm25 = R.build_index(chunks)
    cases = json.load(open(R.GOLDEN, encoding="utf-8"))

    print(f"知识库 {len(set(s for s, _ in chunks))} 份文档 / {len(chunks)} 片段")
    print(f"候选池 {CAND_K} → 重排 → top{R.TOP_K} | 模型 {RERANK_MODEL}")
    print(f"权重目录 {HF_HOME}")

    print("\n嵌入片段与问题（有缓存，重复跑零成本）…")
    vecs = hybrid.embed([t for _, t in chunks], quiet=True)
    qvecs = hybrid.embed([c["question"] for c in cases], quiet=True)

    print("加载重排模型（首次会下载约 1.1GB）…")
    t0 = time.perf_counter()
    model, used = load_reranker()
    print(f"  加载完成 {used}，{time.perf_counter() - t0:.1f}s")

    # 三个模式：前两个是已有基线，第三个是本次的变量。
    # 注意没有做「top10 取前 3 但不重排」这种对照组 —— 那等价于混合 top3，本身就存在。
    def hybrid_fn(b, c, q, v, qv):
        return hybrid.search_hybrid(b, c, q, v, qv)

    def rerank_fn(b, c, q, v, qv):
        cands = hybrid.search_hybrid(b, c, q, v, qv, k=CAND_K)
        return rerank(model, q, cands)

    modes = [
        ("BM25 top3", hybrid.search_bm25),
        ("混合 top3", hybrid_fn),
        (f"混合 top{CAND_K}→重排", rerank_fn),
    ]

    print(f"\n{'模式':<20}{'库内通过':>10}{'top1正确':>10}{'取回字数':>10}{'总耗时':>9}")
    print("-" * 62)
    results = {}
    for name, fn in modes:
        t = time.perf_counter()
        r = hybrid.evaluate(fn, cases, chunks, bm25, vecs, qvecs)
        r["secs"] = time.perf_counter() - t
        results[name] = r
        print(f"{name:<20}{r['ok_in']:>7}/{r['n_in']:<3}{r['top1']:>7}/{r['n_in']:<3}"
              f"{r['vol']:>10}{r['secs']:>8.1f}s")

    base = results["混合 top3"]
    new = results[f"混合 top{CAND_K}→重排"]
    print("\n--- 重排相对混合检索的变化 ---")
    base_fail = {i for i, _ in base["fails"]}
    new_fail = {i for i, _ in new["fails"]}
    print(f"  通过数      {base['ok_in']} → {new['ok_in']}")
    print(f"  top1 正确   {base['top1']} → {new['top1']}")
    print(f"  新修好      {sorted(base_fail - new_fail) or '无'}")
    print(f"  新弄坏      {sorted(new_fail - base_fail) or '无'}")
    print(f"  top1 仍错   {new['top1_miss'] or '无'}")

    # 逐题看排名变化：这是判断「真修好」还是「碰巧」的唯一办法
    print("\n--- 四道已知问题的排名变化 ---")
    qidx = {c["question"]: i for i, c in enumerate(cases)}
    known = [
        ("满多少钱包邮？", "片段级漏召：正确片段原本没进 top3"),
        ("运费险最多能赔多少？", "文档级排错：top1 被工单页抢走"),
        ("运费险一个订单能赔几次？", "措辞差异：一个订单 / 单笔订单"),
        ("同一个问题我已经问了三次了，还需要自助吗？", "措辞差异：三次 / 3 次及以上"),
    ]
    for q, note in known:
        print(f"\n{q}   （{note}）")
        for name, fn in modes:
            hits = fn(bm25, chunks, q, vecs, qvecs[qidx[q]])
            print(f"  {name:<20}" + " | ".join(f"[{k}] {s}" for k, (s, _, _) in enumerate(hits, 1)))

    # 成本：重排每题多花多少时间
    per_q = (new["secs"] - base["secs"]) / max(new["n_in"], 1)
    print(f"\n重排的代价：每题多 {per_q * 1000:.0f}ms（CPU，{len(chunks)} 个片段，候选池 {CAND_K}）")
    print("  接入生产要考虑：这个延迟换 2 道题的 top1，值不值。")


if __name__ == "__main__":
    main()
