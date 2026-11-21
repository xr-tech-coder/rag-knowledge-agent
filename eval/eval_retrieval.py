"""检索层回归门禁 —— 给 CI 用，只跑纯 BM25，零 API 成本。

为什么不跑生成层
---------------
生成层 100 题要真调模型（约 ￥0.37／次），放进每次 push 的 CI 里既贵又慢，
而且没有 key 就跑不了。所以 CI 只守住「不需要模型就能判」的那一层：
切分、召回、top1、取回体积。这四件事恰恰是最容易被无意改坏、
又在界面上看不出来的——改了 CHUNK_SIZE 或 tokenize，答案看着还行，召回已经掉了。

生成层怎么守
-----------
手动触发（workflow_dispatch）跑 eval/smoke10.py，或者本地跑 100 题。
两层的分工写在这里，是为了避免以后有人「为了绿」把生成层也塞进来，
然后因为没配 secret 让 CI 长期红着 —— 长期红的门禁等于没有门禁。

阈值从哪来
---------
2026-11-19 实测（100 题，纯 BM25，CHUNK_SIZE 默认）：
    库内题通过 74/78 ｜ top1 71/78 ｜ 平均取回 965 字

低于基线就是退化，必须失败。允许 0 容差不是苛刻 —— BM25 是确定性的，
同样的数据同样的小版本，跑出来的分数就该一模一样。真要升级依赖导致漂移，
那是需要人看一眼的事，不该被静默放过。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import retrieval as R  # noqa: E402

# 基线（退化即红）
MIN_PASS = 74        # 库内题「来源 + 关键词」同时命中的题数
MIN_TOP1 = 71        # 第 1 名来源就对的题数 —— 刷不了，只能真变好
MAX_CHARS = 1100     # 平均取回字数。涨上去可能是「多给资料把通过率刷高」的假进步
MIN_CHARS = 800      # 掉下去可能是切太碎，关键词被切断


def main():
    args = sys.argv[1:]
    min_pass = int(args[args.index("--min-pass") + 1]) if "--min-pass" in args else MIN_PASS
    min_top1 = int(args[args.index("--min-top1") + 1]) if "--min-top1" in args else MIN_TOP1

    chunks = R.load_chunks()
    bm25 = R.build_index(chunks)
    print(f"知识库：{len(set(s for s, _ in chunks))} 份文档，{len(chunks)} 个片段")

    stats = R.run_golden(bm25, chunks)

    print("\n" + "=" * 60)
    print("门禁结果")
    print("=" * 60)

    checks = [
        ("库内通过", stats["n_pass"], min_pass, ">="),
        ("top1 正确", stats["n_top1"], min_top1, ">="),
        ("平均取回字数", stats["avg_chars"], MIN_CHARS, ">="),
        ("平均取回字数", stats["avg_chars"], MAX_CHARS, "<="),
    ]
    failed = []
    for label, got, limit, op in checks:
        ok = (got >= limit) if op == ">=" else (got <= limit)
        mark = "✅" if ok else "❌"
        print(f"{mark} {label:<12s} {got:>5} {op} {limit}")
        if not ok:
            failed.append(f"{label} {got} {op} {limit}")

    if failed:
        print("\n退化：" + "；".join(failed))
        print("先看是不是切分 / tokenize / top_k 被动过，再看依赖版本有没有漂。")
        return 1

    print(f"\n全部达标（库内 {stats['n_pass']}/{stats['n_in']}，"
          f"应拒答 {stats['n_refuse']} 道不参与判定）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
