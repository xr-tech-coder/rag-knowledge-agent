"""工具层评测：不只看答得对不对，还要看「该调的时候调了没有、不该调的时候忍住了没有」。

为什么不并进那 100 题：
    100 题量的是生成层（说出来的话对不对）。工具题多两个维度 —— 调用行为、以及
    拿到工具数据之后有没有标错出处。混在一个分数里，掉分了你不知道是检索、生成还是调用坏。

判定维度（每道题按需启用）：
    expect_tools_any        至少调用其中一个工具
    expect_no_tool          一个都不许调（误触发检测）
    should_refuse           必须表达「办不了」
    must_contain_any/all    答案里必须出现的字段
    must_match_regex_any    关系式表达（措辞无关，同 golden_set 的 expected_regex）
    must_not_contain        红线：出现即失败
    forbid_citation_numbers 答案里不许出现 [n]（工具数据不带资料编号）

用法：
    TOOLS 会被强制打开，其余（HYBRID / RERANK / MODEL / BASE_URL）沿用环境变量，
    这样测的就是你实际要交付的那套配置。
    python eval_tools.py
"""

import json
import os
import re
import sys
import time

os.environ["TOOLS"] = "1"  # 必须在 import rag 之前：rag 在模块级读这个变量

import rag  # noqa: E402

# 评测不吃缓存。否则同一套题跑第二遍全命中，测出来的是上次的自己。
rag.NO_CACHE = True
import retrieval as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "eval", "tool_golden_set.json")

CITE_RE = re.compile(r"\[\d+\]")


def check(case, answer, called):
    """返回 (通过?, 失败原因列表)。一次把所有问题列全，不要只报第一个。"""
    errs = []

    # ---- 调用行为 ----
    if case.get("expect_no_tool"):
        if called:
            errs.append(f"不该调工具却调了：{called}")
    want = case.get("expect_tools_any") or []
    if want and not (set(want) & set(called)):
        errs.append(f"没调应调的工具（期望其一 {want}，实际 {called or '无'}）")

    # ---- 红线先查，再谈答得对不对（和 judge 同序）----
    bad = [b for b in case.get("must_not_contain", []) if b in answer]
    if bad:
        errs.append(f"出现禁止内容：{bad}")

    if case.get("forbid_citation_numbers"):
        bad_cite = CITE_RE.findall(answer)
        if bad_cite:
            errs.append(f"工具数据被标了资料编号：{bad_cite}")

    if case.get("should_refuse"):
        if not any(w in answer for w in rag.REFUSE_WORDS):
            errs.append("没拒答")

    # ---- 内容 ----
    need_all = case.get("must_contain_all") or []
    miss = [w for w in need_all if w not in answer]
    if miss:
        errs.append(f"缺字段：{miss}")

    need_any = case.get("must_contain_any") or []
    if need_any and not any(w in answer for w in need_any):
        errs.append(f"没命中任何应有表述：{need_any}")

    regexes = case.get("must_match_regex_any") or []
    if regexes:
        flat = answer.translate(rag._KW_STRIP)
        if not any(re.search(p, flat) for p in regexes):
            errs.append(f"没命中关系规则：{regexes}")

    return (not errs), errs


def main():
    if not rag.API_KEY:
        print("没有找到 API_KEY（或 DASHSCOPE_API_KEY），先设一下。")
        return

    rag.CHUNKS = R.load_chunks()
    rag.BM25 = R.build_index(rag.CHUNKS)
    if rag.HYBRID == "1":
        rag.init_vectors()
    if rag.RERANK == "1":
        rag.init_reranker()

    with open(GOLDEN, encoding="utf-8") as f:
        cases = json.load(f)

    import tools as T
    print(f"模型：{rag.MODEL}  检索：{rag.RETRIEVE_MODE}  当前用户：{T.CURRENT_USER}")
    print(f"共 {len(cases)} 题，每题 1~2 次调用。\n")

    silent = lambda _t: None  # 流式输出关掉，只看最终答案
    ok_n = 0
    tot_cost = 0.0
    fails = []
    stat = {}
    call_cnt = 0        # 实际发生工具调用的题数
    bonus_cnt = 0       # 该调工具的题数

    for i, case in enumerate(cases, 1):
        q = case["question"]
        print(f"{i:2d}. {q[:44]}")
        info = rag.ask(q, show_retrieval=False, on_token=silent)
        called = [t["name"] for t in info["tools"]]
        if called:
            call_cnt += 1
        if case.get("expect_tools_any"):
            bonus_cnt += 1

        ok, errs = check(case, info["answer"], called)
        ok_n += ok
        tot_cost += info["cost"]

        cat = case.get("category", "未分类")
        s = stat.setdefault(cat, [0, 0])
        s[1] += 1
        s[0] += 1 if ok else 0

        print(f"    {'✅' if ok else '❌'} 工具={called or '无'} | {info['t_total']:.2f}s | ￥{info['cost']:.5f}")
        for e in errs:
            print(f"       - {e}")
        print(f"       答：{info['answer'][:110].replace(chr(10), ' ')}")
        if not ok:
            fails.append((i, q, errs, info["answer"]))
        print()

    print("=" * 62)
    print(f"工具层准确率：{ok_n}/{len(cases)} ({ok_n / len(cases) * 100:.0f}%)")
    print(f"总花费 ￥{tot_cost:.4f} | 单题均价 ￥{tot_cost / len(cases):.5f}")
    print(f"该调工具的 {bonus_cnt} 题，实际发生调用 {call_cnt} 题"
          f"（含误触发，仅作参考）")

    print("\n分层：")
    for cat, (a, b) in stat.items():
        print(f"  {cat:<8s} {a}/{b}")

    if fails:
        print("\n失败清单：")
        for i, q, errs, ans in fails:
            print(f"  {i}. {q}")
            for e in errs:
                print(f"     - {e}")
            print(f"     实际：{ans[:140].replace(chr(10), ' ')}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
