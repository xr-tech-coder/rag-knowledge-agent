"""实验 4：BM25 vs 向量 vs 混合检索（RRF 融合）。

为什么单独一个文件：基线 retrieval.py 一行不动，对比才干净。
三种模式跑同一批 50 题，只改「怎么召回」，其余全固定。

跑法：
    .venv\\Scripts\\python.exe hybrid.py                三种模式全部对比
    .venv\\Scripts\\python.exe hybrid.py "满多少钱包邮"   单题看三种模式的召回

需要的环境变量（本机已有 API_KEY）：
    EMBED_BASE_URL  走 OpenAI 兼容的 /embeddings，默认取 BASE_URL
    EMBED_MODEL     默认 text-embedding-v4

向量会缓存到 eval/.vec_cache.npz，同一段文本只花一次钱。
"""
import hashlib
import json
import os
import sys

import numpy as np
from openai import OpenAI

import retrieval as R

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v4")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL") or os.getenv("BASE_URL") or ""
CACHE = os.path.join(R.HERE, "eval", ".vec_cache.npz")

RRF_K = 60      # 标准值，来自原论文；作用是压低头部排名的绝对优势
TOP_K = R.TOP_K


# ---------------- 向量 ----------------
def _client():
    """拿 embedding 客户端。

    注意这里抛 RuntimeError 而不是 sys.exit()：这个模块会被 rag.py 引用，
    配置缺失时应该让调用方有机会降级回 BM25，而不是把整个进程杀掉。
    """
    key = os.getenv("API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("缺少 API_KEY / DASHSCOPE_API_KEY，无法调用 embedding")
    if not EMBED_BASE_URL:
        raise RuntimeError(
            "缺少 EMBED_BASE_URL（或 BASE_URL）。注意默认的 deepseek 端点没有 embedding 接口。"
        )
    return OpenAI(api_key=key, base_url=EMBED_BASE_URL)


def _key(text):
    return hashlib.sha1((EMBED_MODEL + "|" + text).encode("utf-8")).hexdigest()[:20]


def embed(texts, quiet=False):
    """返回 (n, d) 矩阵。带本地缓存，同一段文本不会重复计费。"""
    cache = {}
    if os.path.exists(CACHE):
        with np.load(CACHE) as z:
            cache = {k: z[k] for k in z.files}

    todo = [t for t in texts if _key(t) not in cache]
    if todo:
        client = _client()
        # 一次最多 10 条，官方限制；分批更稳
        for i in range(0, len(todo), 10):
            batch = todo[i:i + 10]
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            for text, item in zip(batch, resp.data):
                cache[_key(text)] = np.asarray(item.embedding, dtype=np.float32)
            if not quiet:
                print(f"  嵌入 {min(i + 10, len(todo))}/{len(todo)} 条（新调用）")
        np.savez(CACHE, **cache)
    elif not quiet:
        print(f"  全部命中缓存（{len(texts)} 条），本次零成本")

    return np.vstack([cache[_key(t)] for t in texts])


def cosine(vecs, qv):
    """归一化后点积 = 余弦相似度。"""
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    n[n == 0] = 1e-9
    return (vecs / n) @ (qv / (np.linalg.norm(qv) + 1e-9))


# ---------------- 三种召回 ----------------
def search_bm25(bm25, chunks, q, vecs, qv):
    return R.search(bm25, chunks, q)


def search_vec(bm25, chunks, q, vecs, qv):
    scores = cosine(vecs, qv)
    order = np.argsort(-scores)[:TOP_K]
    return [(chunks[i][0], float(scores[i]), chunks[i][1]) for i in order]


def search_hybrid(bm25, chunks, q, vecs, qv, k=None):
    """RRF：只用排名，不用分数。

    为什么不直接加权平均：BM25 分数是无界的（0~30+），余弦是 0~1，
    两者量纲不同，加权系数就是硬凑的超参，换个数据集就失效。
    RRF 只看「在各自榜单里排第几」，天然免疫量纲问题。

    k 默认取 TOP_K。重排实验要先把候选池放大到 10，所以留了这个参数。
    """
    s_bm = np.asarray(bm25.get_scores(R.tokenize(q)), dtype=float)
    s_vc = cosine(vecs, qv)

    fused = np.zeros(len(chunks), dtype=float)
    for scores in (s_bm, s_vc):
        for rank, idx in enumerate(np.argsort(-scores)):
            fused[idx] += 1.0 / (RRF_K + rank + 1)

    order = np.argsort(-fused)[:k or TOP_K]
    return [(chunks[i][0], float(fused[i]), chunks[i][1]) for i in order]


MODES = {
    "BM25（基线）": search_bm25,
    "纯向量": search_vec,
    "混合 RRF": search_hybrid,
}


# ---------------- 评测 ----------------
def evaluate(search_fn, cases, chunks, bm25, vecs, qvecs):
    n_in = ok_in = n_out = 0
    top1_hit = kw_right = vol = 0
    fails = []
    top1_miss = []      # top1 排错了，但靠 top2/top3 兜住 —— 排序问题，不是召回问题
    out_scores = []

    for i, case in enumerate(cases, 1):
        hits = search_fn(bm25, chunks, case["question"], vecs, qvecs[i - 1])
        top1 = hits[0][1] if hits else 0.0

        if case.get("should_refuse"):
            if case.get("category") in ("越权", "注入"):
                continue
            n_out += 1
            # 库外题在检索层「判不了」：BM25 分数和余弦相似度量纲不同，
            # 阈值只对 BM25 有意义。这里只留分数做观察，不给通过/失败。
            out_scores.append(top1)
            continue

        n_in += 1
        want_src = case.get("expected_sources", [])
        got_src = [s for s, _, _ in hits]
        got_text = " ".join(t for _, _, t in hits)
        src_ok = all(s in got_src for s in want_src)

        if case.get("expected_keyword_any"):
            kws = case["expected_keyword_any"]
            kw_ok = any(R.kw_hit(k, got_text) for k in kws)
        else:
            kw = case.get("expected_keyword", "")
            kws = [kw] if kw else []
            kw_ok = R.kw_hit(kw, got_text) if kw else True

        top1_hit += hits[0][0] in want_src
        kw_right += (any(R.kw_hit(k, " ".join(t for s, _, t in hits if s in want_src)) for k in kws)
                     if kws else 1)
        vol += sum(len(t) for _, _, t in hits)

        if src_ok and kw_ok:
            ok_in += 1
            if hits[0][0] not in want_src:
                top1_miss.append(i)
        else:
            fails.append((i, ("来源✗" if not src_ok else "") + ("关键词✗" if not kw_ok else "")))

    return dict(n_in=n_in, ok_in=ok_in, top1=top1_hit, kw_right=kw_right,
                n_out=n_out, out_scores=out_scores, vol=vol // max(n_in, 1),
                fails=fails, top1_miss=top1_miss)


def main():
    chunks = R.load_chunks()
    bm25 = R.build_index(chunks)
    cases = json.load(open(R.GOLDEN, encoding="utf-8"))

    print(f"知识库 {len(set(s for s, _ in chunks))} 份文档 / {len(chunks)} 片段 | "
          f"embedding {EMBED_MODEL} | RRF k={RRF_K}")

    print("嵌入片段…")
    vecs = embed([t for _, t in chunks])
    print("嵌入问题…")
    qvecs = embed([c["question"] for c in cases])

    print(f"\n{'模式':<14}{'库内通过':>10}{'top1正确':>10}{'词在对源':>10}{'取回字数':>10}  失败题号")
    print("-" * 76)
    results = {}
    for name, fn in MODES.items():
        r = evaluate(fn, cases, chunks, bm25, vecs, qvecs)
        results[name] = r
        nums = ",".join(str(f[0]) for f in r["fails"]) or "无"
        print(f"{name:<14}{r['ok_in']:>7}/{r['n_in']:<3}{r['top1']:>7}/{r['n_in']:<3}"
              f"{r['kw_right']:>7}/{r['n_in']:<3}{r['vol']:>10}  {nums}")

    base = results["BM25（基线）"]["fails"]
    print("\n--- 相对基线的变化 ---")
    for name in ("纯向量", "混合 RRF"):
        r = results[name]
        fixed = [i for i, _ in base if i not in [j for j, _ in r["fails"]]]
        broke = [i for i, _ in r["fails"] if i not in [j for j, _ in base]]
        print(f"{name}: 修好 {fixed or '无'} | 新坏 {broke or '无'}")

    print("\n--- top1 排错但靠 top2/top3 兜住的题（排序问题）---")
    for name in MODES:
        m = results[name]["top1_miss"]
        print(f"  {name:<12}{m or '无'}")

    print("\n--- 库外题 top1 分数（仅观察，量纲不同不可比）---")
    for name in MODES:
        s = results[name]["out_scores"]
        print(f"  {name:<12}" + " ".join(f"{x:6.2f}" for x in s))

    print("\n--- 两个动机题的实际排名 ---")
    qidx = {c["question"]: i for i, c in enumerate(cases)}
    for q in ("满多少钱包邮？", "运费险最多能赔多少？"):
        print(f"\n{q}")
        for name, fn in MODES.items():
            hits = fn(bm25, chunks, q, vecs, qvecs[qidx[q]])
            print(f"  {name:<12}" + " | ".join(f"[{k}] {s} {sc:.4f}" for k, (s, sc, _) in enumerate(hits, 1)))


if __name__ == "__main__":
    main()
