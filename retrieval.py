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

# 【已废止】REFUSE_THRESHOLD（曾 = 3.0）—— 想用「检索最高分低于阈值就拒答」。
# 实测（20 个片段、纯 BM25）：
#     库内题最低分 2.14，库外题最高分 2.85，区分度 = -0.71（负数），库外题分数反而更高。
# 补到 100 题后跨度扩大到 0.00 ~ 3.13，更没区分力。
# 结论：纯 BM25 + 小数据量下，「用绝对分数判拒答」这条路是死的，不是调参能救的。
# 拒答已完全交给生成层（prompt 规则 2/5/6 + rag.py 的 judge）。
# 这个常量只留在注释里当记录，代码中不再引用 —— 免得有人以为它还在生效。

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


# 与 rag.py 的 _KW_STRIP / _kw_hit 同一套口径：空白和纯排版标点不承载语义。
# 两层标准必须一致，否则会出现「检索层判过、生成层判不过」，排查时互相打架。
_KW_STRIP = str.maketrans("", "", " \t\n\r、，,。．；;：:（）()【】[]·\"'“”‘’")


def kw_hit(keyword, text):
    """关键词是否命中：先按字面比，不中就删掉空白和排版标点再比。"""
    return keyword in text or keyword.translate(_KW_STRIP) in text.translate(_KW_STRIP)


def show(question, hits):
    print(f"\n问题：{question}")
    for rank, (src, score, text) in enumerate(hits, 1):
        head = " ".join(text.split())[:60]
        print(f"  [{rank}] {src:<26s} 分数 {score:6.2f}  {head}…")


def run_golden(bm25, chunks):
    with open(GOLDEN, encoding="utf-8") as f:
        cases = json.load(f)

    n_in = n_in_ok = 0          # 库内题（有标准答案）
    n_top1_ok = 0               # 其中「第 1 名来源就对」的题数
    vol_total = 0               # 库内题取回文本的总字数，用来算平均
    n_out = 0                   # 应拒答题总数（三类合计，不判对错，只统计）
    in_min_top1 = None          # 库内题里最低的最高分，用来对照阈值是否合理
    out_top1 = []               # 库外题的分数，用来算区分度

    for i, case in enumerate(cases, 1):
        q = case["question"]
        hits = search(bm25, chunks, q)
        top1 = hits[0][1] if hits else 0.0

        # ---------- 应拒答题：检索层一律不判对错，只报分数 ----------
        # 为什么三类都不判：
        #   · 越权、注入：资料里本来就有「订单」「退款」这些词，注入题的毒内容更是
        #     故意埋进文档的 —— 分数高是必然结果。对错要看生成层有没有真的执行那个动作。
        #   · 库外：已实测区分度 -0.71 ~ -0.99（负数），「用绝对分数判库里有没有」是死的。
        #     既然这把尺子自己都被否掉了，就不该再拿它给库外题打 ❌。
        # 分数照报，是为了留一条线：以后换向量/重排，库外题的分数应该整体往下走。
        if case.get("should_refuse"):
            cat = case.get("category", "")
            n_out += 1
            if cat == "库外":
                out_top1.append(top1)   # 只有纯库外题的分数可作对照
            print(f"{i:2d}. ➖ {q[:30]:<32s} {cat}题 最高分 {top1:5.2f}  检索层不判定")
            continue

        # ---------- 库内题：来源全中 + 关键词命中 ----------
        n_in += 1
        want_src = case.get("expected_sources", [])

        got_src = [s for s, _, _ in hits]
        got_text = " ".join(t for _, _, t in hits)
        vol_total += len(got_text)

        # 用 all() 而不是任何一道命中：多跳题必须两份文档都取到才算过
        src_ok = all(s in got_src for s in want_src)

        # top1 正确 = 第 1 名的来源就在期望来源里。
        # 这个指标替代不了「通过」，因为多跳题要求两份文档同时在场；
        # 但它测的是排序质量，而排序质量没法靠「多给资料」刷高（第 1 名只有一个）。
        top1_ok = bool(want_src) and hits[0][0] in want_src
        n_top1_ok += top1_ok

        # 关键词两种写法，与 rag.py 的 judge() 保持同一套口径：
        #   expected_keyword      单值，必须原样出现
        #   expected_keyword_any  列表，命中任一即可（同一个答案的不同说法）
        if case.get("expected_keyword_any"):
            want_kw = " / ".join(case["expected_keyword_any"])
            kw_ok = any(kw_hit(k, got_text) for k in case["expected_keyword_any"])
        else:
            want_kw = case.get("expected_keyword", "")
            kw_ok = kw_hit(want_kw, got_text) if want_kw else True

        ok = src_ok and kw_ok
        n_in_ok += ok

        if in_min_top1 is None or top1 < in_min_top1:
            in_min_top1 = top1

        flag = "✅" if ok else "❌"
        # 来源进了 top3 却没排第一：能答对是靠模型自己往后翻，换个模型就未必
        warn = "  ⚠ 进了没排对" if (src_ok and not top1_ok) else ""
        print(f"{i:2d}. {flag} {q[:30]:<32s} 来源{'✓' if src_ok else '✗'} "
              f"关键词{'✓' if kw_ok else '✗'} top1{'✓' if top1_ok else '✗'}{warn}")
        if not src_ok:
            print(f"     期望 {want_src}，实际 {got_src}")
        if not kw_ok:
            print(f"     期望关键词「{want_kw}」不在取回的 {TOP_K} 段里")

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print(f"库内题：{n_in_ok}/{n_in} 通过 ({n_in_ok / n_in * 100:.0f}%)")
    if n_in:
        print(f"top1 正确：{n_top1_ok}/{n_in} ({n_top1_ok / n_in * 100:.0f}%) "
              f"| 平均取回 {vol_total // n_in} 字")
        print("   ↑ 「通过」能被多给资料刷高（字多，关键词更容易撞上）；")
        print("     top1 正确率刷不了（第 1 名只有一个）。三个数一起看，才分得清真变好还是换代价。")

    # 应拒答题不给通过率：检索层判不了，硬算出来的数字只会误导
    if n_out:
        print(f"\n应拒答题 {n_out} 道（含越权/注入），检索层不判对错。")
        if out_top1:
            scores = " ".join(f"{x:.2f}" for x in sorted(out_top1, reverse=True))
            print(f"  其中纯库外题 {len(out_top1)} 道最高分：{scores}")
            print("  ↑ 这个跨度就是「分数分不出库里有没有」的证据。加向量/重排后这几个数应该整体下降，")
            print("    那才叫检索层学会了「不知道」。不要把它们换算成任何通过率。")

    if in_min_top1 is not None and out_top1:
        gap = in_min_top1 - max(out_top1)
        print(f"\n区分度：库内题最低分 {in_min_top1:.2f} − 库外题最高分 {max(out_top1):.2f} = {gap:.2f}")
        if gap < 1.0:
            print("⚠ 分不开。分数在「库里有」和「库里没有」之间没有缝隙，")
            print("  得靠向量检索 + 重排把分布拉开，或干脆交给生成层判断。别再调阈值了。")
        else:
            print("  有缝隙，分数可以当辅助信号（主判定仍交给生成层）。")
    print("\n上面三个数（通过率 / top1 / 平均取回字数）就是你的基线。")
    print("以后每改一处（切分/加向量/加重排），回来跑一次对比。")


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
