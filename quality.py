"""检索层质量基线：阈值与判定逻辑只有这一份。

谁在读它
--------
- `eval/eval_retrieval.py` —— CI 每次 push 跑，不达标退出码 1
- `api.py` 的 `GET /eval/retrieval` —— 线上随时能查「现在这版还达标吗」

为什么要抽出来：两边各写一份判定，迟早出现「CI 说达标、接口说不达标」，
到那时没人知道该信哪个。阈值被复制的那天，它就不再是基线了。

阈值哪来的
---------
2026-11-19 实测（100 题，纯 BM25，CHUNK_SIZE 默认）：
    库内通过 74/78 ｜ top1 71/78 ｜ 平均取回 965 字

BM25 是确定性的：同数据、同小版本，分数应当一模一样。所以容差是 0 ——
真要漂了（升级依赖、改切分），那是需要人看一眼的事，不该被静默放过。

三个数为什么都要看
----------------
「通过」能被多给资料刷高（片段切大、字变多，关键词更容易撞上），
top1 刷不了（第 1 名只有一个），取回字数则直接暴露「是不是靠塞字刷的分」。
只卡通过率，等于允许一种看起来很努力的退化。
"""

BASELINE = {
    "min_pass": 74,     # 库内题「来源 + 关键词」同时命中
    "min_top1": 71,     # 第 1 名来源就对
    "min_chars": 800,   # 平均取回字数下限：切太碎，关键词会被切断
    "max_chars": 1100,  # 上限：涨太多说明在靠多塞字刷通过率
}


def evaluate(stats, baseline=None):
    """按基线判定一次检索层结果。

    stats: retrieval.run_golden() 的返回值
    baseline: 覆盖个别阈值（评测脚本用 --min-pass 之类的参数时）

    返回 {"ok": bool, "checks": [...], "failed": [...]}
    """
    b = dict(BASELINE)
    if baseline:
        b.update(baseline)

    checks = [
        ("库内通过", stats.get("n_pass", 0), ">=", b["min_pass"]),
        ("top1 正确", stats.get("n_top1", 0), ">=", b["min_top1"]),
        # 上下限分开命名：同名会出现两条一模一样的失败行，
        # 失败时看不出是「切太碎」还是「切太大」，而那正是这条门禁要抓的东西
        ("取回字数下限", stats.get("avg_chars", 0), ">=", b["min_chars"]),
        ("取回字数上限", stats.get("avg_chars", 0), "<=", b["max_chars"]),
    ]

    result = []
    failed = []
    for name, got, op, limit in checks:
        ok = (got >= limit) if op == ">=" else (got <= limit)
        result.append({"name": name, "got": got, "op": op, "limit": limit, "ok": ok})
        if not ok:
            failed.append(f"{name} {got} {op} {limit}")

    return {"ok": not failed, "checks": result, "failed": failed}
