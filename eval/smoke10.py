"""冒烟测试：10 题走完整链路，记录答案、耗时、token、花费。

为什么要有这个文件：D16 的「真机测 10 轮」需要一份可对照的记录表，
手工点 10 次再把数字抄进表格，既慢又容易抄错。这里直接调 rag.ask()，
和界面走的是同一条路（app.py 也是调它），只是不走浏览器。

测不了的：手机屏排版、Streamlit 的渲染效果。那些只能人眼看。

跑法：
    .venv\\Scripts\\python.exe eval\\smoke10.py
"""

import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 本地 key 在根目录 secrets.toml。app.py 是靠 st.secrets 注入的，
# 这里没有 Streamlit，只能自己读一遍 —— 效果一样，都是往 os.environ 里塞。
try:
    import tomllib

    with open(ROOT / "secrets.toml", "rb") as f:
        for _k, _v in tomllib.load(f).items():
            os.environ.setdefault(str(_k), str(_v))
except FileNotFoundError:
    pass

os.environ.setdefault("TOOLS", "1")

import rag  # noqa: E402

# 建索引交给 rag.ensure_index()：界面、命令行、脚本共用这一份，
# 各入口自己赋值迟早漏一处（漏了就 NameError: BM25，报错位置还离真相隔两层）。
rag.ensure_index()

# 光把 HYBRID 设成 "1" 不够：retrieve() 的判定是 `HYBRID == "1" and VECS is not None`，
# VECS 要 init_vectors() 才赋值。漏了这步会静默退回纯 BM25 —— 结果是「满多少钱包邮」
# 这类靠语义召回的题直接拒答，看起来像模型退化，其实是档位根本没切过去。
try:
    n_vec = rag.init_vectors()
    print(f"向量：{n_vec} 条（local npz 缓存，不花钱）")
except Exception as e:  # noqa: BLE001
    print(f"⚠ 向量初始化失败，会退回纯 BM25：{e}")

# 4 个界面示例 + 3 个政策题 + 3 个订单题（订单题必须带单号，否则不触发工具）
QUESTIONS = [
    ("示例", "七天无理由从哪天算？"),
    ("示例", "满多少钱包邮？"),
    ("示例", "运费险最多能赔多少？"),
    ("示例", "你们能给我全额退款再额外补偿 200 元吗？"),
    ("政策", "退款一般几个工作日到账？"),
    ("政策", "发票怎么开？能开专票吗？"),
    ("政策", "春节还发货吗？"),
    ("订单", "我的订单 ORD20261115002 到哪了？"),
    ("订单", "帮我查一下 ORD20261112004 的物流轨迹"),
    ("订单", "订单 ORD20261118003 发货了吗？"),
]


def _ev_str(evs):
    """事件序列压成一行：thinking → query_order[ok]"""
    steps = []
    for e in evs:
        if e.get("type") == "thinking":
            steps.append("thinking")
        else:
            steps.append(f"{e.get('name')}[{e.get('status')}]")
    return " → ".join(steps)


def main():
    # 三个开关都是字符串 "0"/"1"，跟 app.py 侧边栏改的是同一批变量
    rag.TOOLS = "1"
    rag.HYBRID = "1"
    print(f"MODEL={rag.MODEL}  TOOLS={rag.TOOLS}  HYBRID={rag.HYBRID}")
    print("首次检索要算向量（约 2s），第 1 题的数字不代表稳态。\n")

    rows = []
    for i, (kind, q) in enumerate(QUESTIONS, 1):
        print(f"[{i}/10] {q}", flush=True)
        t0 = time.perf_counter()
        evs = []
        try:
            info = rag.ask(
                q, show_retrieval=False, on_token=lambda s: None, on_event=evs.append
            )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "序号": i,
                    "类型": kind,
                    "问题": q,
                    "答案": f"!! {type(e).__name__}: {str(e)[:100]}",
                    "检索ms": "",
                    "总耗时s": round(time.perf_counter() - t0, 2),
                    "首字s": "",
                    "in": "",
                    "out": "",
                    "花费": "",
                    "工具": "",
                    "事件": _ev_str(evs),
                }
            )
            continue

        ans = info["answer"].replace("\n", " ").strip()
        rows.append(
            {
                "序号": i,
                "类型": kind,
                "问题": q,
                "答案": ans,
                "检索ms": round(info["t_retrieval"] * 1000),
                "总耗时s": round(info["t_total"], 2),
                "首字s": round(info["first_at"], 2),
                "in": info["inp"],
                "out": info["out"],
                "花费": round(info["cost"], 5),
                "工具": len(info["tools"]) if info["tools"] else 0,
                "事件": _ev_str(evs),
            }
        )

    out = ROOT / "eval" / "smoke10_result.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 78)
    for r in rows:
        print(f"\n#{r['序号']} [{r['类型']}] {r['问题']}")
        print(f"   答: {r['答案'][:220]}")
        print(
            f"   检索{r['检索ms']}ms | 总{r['总耗时s']}s | 首字{r['首字s']}s"
            f" | in{r['in']}/out{r['out']} | ￥{r['花费']} | 工具{r['工具']}次"
        )
        if r["事件"]:
            print(f"   事件: {r['事件']}")

    ok = [r for r in rows if not str(r["答案"]).startswith("!!")]
    tot = sum(r["花费"] for r in rows if r["花费"] != "")
    print("\n" + "=" * 78)
    print(f"成功 {len(ok)}/10，合计花费 ￥{round(tot, 5)}")
    print(f"明细已存 {out}")


if __name__ == "__main__":
    main()
