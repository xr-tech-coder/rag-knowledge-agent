"""
D2：检索。给一个问题，从 data/ 里找出最相关的几段，打印来源和分数。

跑法：
    .venv\\Scripts\\python.exe retrieval.py                 交互模式，随便问
    .venv\\Scripts\\python.exe retrieval.py 七天无理由    直接问一句
    .venv\\Scripts\\python.exe retrieval.py --golden      批量跑 eval/golden_set.json

这一版只用 BM25（关键词匹配），故意不上向量。
原因：先看纯关键词能做到什么程度，后面加向量时才有对照。
"""
import glob
import json
import os
import sys

from rank_bm25 import BM25Okapi

CHUNK_SIZE = 400      # 每段字数
OVERLAP = 80          # 相邻段重叠字数，防止一句话被切断
TOP_K = 3             # 取前几段

# 拒答阈值：检索最高分低于这个数，就认为「库里没有能回答的资料」，应该拒答。
# ⚠ 这个值很脆弱，实测依据（19 个片段、BM25 纯关键词）：
#     库内题最低分 3.06（第 3 题，多跳题）
#     库外题最高分 2.70（第 11 题）
#   两者只差 0.36 —— 阈值往左挪一点，库外题被当成有答案；往右挪一点，库内题被拒答。
#   这说明「用绝对分数判拒答」在小数据量 + 纯 BM25 下不可靠。
#   真正的解法是后面加向量检索 + 重排，用重排分数判（分数分布拉得开），或交给生成层判断。
REFUSE_THRESHOLD = 3.0

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
GOLDEN = os.path.join(HERE, "eval", "golden_set.json")


def tokenize(text):
    """中文没有空格，BM25 需要先切成词。

    这里用「字符二元组」(bi-gram)：相邻两个字算一个词。
    例："七天无理由" → 七天 / 天无 / 无理 / 理由
    好处：不用装分词库，中文效果够用。
    代价：语义靠字面重合，同义改写（"包邮" vs "免运费"）匹配不上 —— 这正是后面要加向量的理由。
    """
    text = "".join(text.split())
    return [text[i:i + 2] for i in range(len(text) - 1)] or [text]


def load_chunks():
    """把 data/ 下所有 md 切段，返回 [(来源文件名, 片段文本)]。"""
    chunks = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        name = os.path.basename(path)
        start = 0
        while start < len(text):
            piece = text[start:start + CHUNK_SIZE]
            if piece.strip():
                chunks.append((name, piece))
            start += CHUNK_SIZE - OVERLAP
    return chunks


def build_index(chunks):
    corpus = [tokenize(t) for _, t in chunks]
    return BM25Okapi(corpus)


def search(bm25, chunks, question, top_k=TOP_K):
    scores = bm25.get_scores(tokenize(question))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i][0], scores[i], chunks[i][1]) for i in order]


def show(question, hits):
    print(f"\n问题：{question}")
    for rank, (src, score, text) in enumerate(hits, 1):
        head = " ".join(text.split())[:60]
        print(f"  [{rank}] {src:<26s} 分数 {score:6.2f}  {head}…")


def run_golden(bm25, chunks):
    with open(GOLDEN, encoding="utf-8") as f:
        cases = json.load(f)

    n_in = n_in_ok = 0          # 库内题（有标准答案）
    n_out = n_out_ok = 0        # 库外题（应当拒答）
    in_min_top1 = None          # 库内题里最低的最高分，用来对照阈值是否合理
    out_top1 = []

    for i, case in enumerate(cases, 1):
        q = case["question"]
        hits = search(bm25, chunks, q)
        top1 = hits[0][1] if hits else 0.0

        # ---------- 应拒答题 ----------
        # 检索层只能判「库外题」（资料真没有，最高分低才算对）。
        # 「越权」「注入」两类判不了：越权题的资料常见词（订单/退款）本来就在库里，
        # 注入题更是故意把毒内容藏进了文档 —— 分数高是必然的，对错要看生成层
        # 有没有真的执行那个动作（改地址、念内部备注、承诺赔偿）。
        if case.get("should_refuse"):
            cat = case.get("category", "")
            if cat in ("越权", "注入"):
                print(f"{i:2d}. ➖ {q[:30]:<32s} {cat}题 最高分 {top1:5.2f}  检索层不判定")
                continue

            n_out += 1
            out_top1.append(top1)
            ok = top1 < REFUSE_THRESHOLD
            n_out_ok += ok
            flag = "✅" if ok else "❌"
            verdict = "正确拒答" if ok else "最高分够高 → 可能照着无关片段瞎答"
            print(f"{i:2d}. {flag} {q[:30]:<32s} 库外题 最高分 {top1:5.2f}  {verdict}")
            continue

        # ---------- 库内题：来源全中 + 关键词命中 ----------
        n_in += 1
        want_src = case.get("expected_sources", [])
        want_kw = case.get("expected_keyword", "")

        got_src = [s for s, _, _ in hits]
        got_text = " ".join(t for _, _, t in hits)

        # 用 all() 而不是任何一道命中：多跳题必须两份文档都取到才算过
        src_ok = all(s in got_src for s in want_src)
        kw_ok = (want_kw in got_text) if want_kw else True
        ok = src_ok and kw_ok
        n_in_ok += ok

        if in_min_top1 is None or top1 < in_min_top1:
            in_min_top1 = top1

        flag = "✅" if ok else "❌"
        print(f"{i:2d}. {flag} {q[:32]:<34s} 来源{'✓' if src_ok else '✗'} 关键词{'✓' if kw_ok else '✗'}")
        if not src_ok:
            print(f"     期望 {want_src}，实际 {got_src}")
        if not kw_ok:
            print(f"     期望关键词「{want_kw}」不在取回的 {TOP_K} 段里")

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print(f"库内题：{n_in_ok}/{n_in} 通过 ({n_in_ok / n_in * 100:.0f}%)")
    if n_out:
        print(f"拒答题：{n_out_ok}/{n_out} 正确拒答 ({n_out_ok / n_out * 100:.0f}%)")

    if in_min_top1 is not None and out_top1:
        gap = in_min_top1 - max(out_top1)
        print(f"\n拒答阈值 {REFUSE_THRESHOLD} | 库内题最低分 {in_min_top1:.2f} | 库外题最高分 {max(out_top1):.2f}")
        if gap < 1.0:
            print(f"⚠ 区分度只有 {gap:.2f} 分 —— 阈值不可靠。这个区间里靠分数分不开「库里有」和「库里没有」，")
            print("  必须靠向量检索 + 重排把分数分布拉开，或交给生成层判断。别在这里继续调阈值。")
        else:
            print(f"  区分度 {gap:.2f} 分，阈值有参考价值。")
    print("\n这两个数就是你的基线。以后每改一处（切分/加向量/加重排），回来跑一次对比。")


def main():
    chunks = load_chunks()
    bm25 = build_index(chunks)
    print(f"知识库：{len(set(s for s, _ in chunks))} 份文档，{len(chunks)} 个片段")

    args = [a for a in sys.argv[1:]]

    if "--golden" in args:
        run_golden(bm25, chunks)
        return

    if args:
        show(" ".join(args), search(bm25, chunks, " ".join(args)))
        return

    print("输入问题，回车看结果；直接回车退出。")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        show(q, search(bm25, chunks, q))


if __name__ == "__main__":
    main()
