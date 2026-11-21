"""降级链的离线测试：不打模型、不花钱，全用桩。

额度断了也能跑。四件事必须成立：
  1. 同一问题第二次走缓存，花费清零
  2. 配置变了（指纹变化）缓存不命中
  3. 服务故障（403）→ 命中预设话术，且答案明确标注是标准答复
  4. 代码 bug（NameError）→ 必须原样抛出，不被降级掩盖

跑法：
    .venv\\Scripts\\python.exe eval\\test_degrade.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402

rag.TOOLS = "0"
rag._CACHE.clear()

FAILS = []


def check(name, cond, detail=""):
    print(f"{'✓' if cond else '✗'} {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _ok_info(answer="测试答案", cost=0.004):
    return {
        "answer": answer,
        "hits": [],
        "mode": "hybrid",
        "tools": [],
        "t_retrieval": 0.01,
        "first_at": 0.5,
        "t_total": 1.2,
        "inp": 100,
        "out": 20,
        "cost": cost,
    }


def fake_ok(question, show_retrieval, emit, t0):
    emit("测试答案")
    return _ok_info()


def fake_403(question, show_retrieval, emit, t0):
    # 模仿真实异常：类型名是 PermissionDeniedError，消息里带 403
    raise type("PermissionDeniedError", (Exception,), {})(
        "Error code: 403 - Free quota exhausted"
    )


def fake_bug(question, show_retrieval, emit, t0):
    raise NameError("name 'BM25' is not defined")


# ---------- 1. 缓存命中 ----------
rag._ask_basic = fake_ok
rag._CACHE.clear()

first = rag.ask("满多少钱包邮？")
second = rag.ask("满多少钱包邮？")
check("第一次不走缓存", first.get("degraded") is None)
check("第二次命中缓存", second.get("degraded") == "cache", str(second.get("degraded")))
check("缓存命中后花费归零", second["cost"] == 0 and second["t_total"] == 0)
check("缓存命中仍然回吐正文", second["answer"] == "测试答案")

# ---------- 2. 指纹变化后缓存失效 ----------
rag.HYBRID = "0" if rag.HYBRID == "1" else "1"
third = rag.ask("满多少钱包邮？")
check("改配置后缓存不命中", third.get("degraded") is None, str(third.get("degraded")))

# ---------- 3. 服务故障 → 预设话术 ----------
rag._ask_basic = fake_403
rag._CACHE.clear()

canned = rag.ask("满多少钱包邮？")
check("403 降级到预设话术", canned.get("degraded") == "canned", str(canned.get("degraded")))
check("答案标注了不是实时生成", "不是实时生成" in canned["answer"])
check("答案含标准答复正文", "99 元包邮" in canned["answer"])
check("降级结果不写进缓存", rag.ask("满多少钱包邮？").get("degraded") == "canned")

# ---------- 4. 服务故障 + 无预设 → 转人工 ----------
handoff = rag.ask("你们老板的电话是多少？")
check("无预设时转人工", handoff.get("degraded") == "handoff", str(handoff.get("degraded")))
check("转人工答案标注了故障", "自动答复暂时不可用" in handoff["answer"])

# ---------- 5. 代码 bug 不许被降级 ----------
rag._ask_basic = fake_bug
rag._CACHE.clear()

try:
    rag.ask("满多少钱包邮？")
    check("NameError 必须抛出", False, "被吞掉了")
except NameError:
    check("NameError 原样抛出", True)

print()
if FAILS:
    print(f"失败 {len(FAILS)} 项：{FAILS}")
    sys.exit(1)
print("全部通过")
