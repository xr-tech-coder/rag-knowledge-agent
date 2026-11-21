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

阈值在 quality.py，和线上 /eval/retrieval 读的是同一份。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quality  # noqa: E402
import retrieval as R  # noqa: E402


def main():
    args = sys.argv[1:]
    override = {}
    if "--min-pass" in args:
        override["min_pass"] = int(args[args.index("--min-pass") + 1])
    if "--min-top1" in args:
        override["min_top1"] = int(args[args.index("--min-top1") + 1])

    chunks = R.load_chunks()
    bm25 = R.build_index(chunks)
    print(f"知识库：{len(set(s for s, _ in chunks))} 份文档，{len(chunks)} 个片段")

    stats = R.run_golden(bm25, chunks)
    verdict = quality.evaluate(stats, override)

    print("\n" + "=" * 60)
    print("门禁结果")
    print("=" * 60)
    for c in verdict["checks"]:
        print(f"{'✅' if c['ok'] else '❌'} {c['name']:<12s} {c['got']:>5} {c['op']} {c['limit']}")

    if not verdict["ok"]:
        print("\n退化：" + "；".join(verdict["failed"]))
        print("先看是不是切分 / tokenize / top_k 被动过，再看依赖版本有没有漂。")
        return 1

    print(f"\n全部达标（库内 {stats['n_pass']}/{stats['n_in']}，"
          f"应拒答 {stats['n_refuse']} 道不参与判定）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
