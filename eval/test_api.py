"""HTTP 服务层的离线测试：不打模型、不花钱，全用桩。

为什么值得单独测这一层：界面走的是"回调 + 即时渲染"，API 走的是"状态码 + JSON"。
同一份 rag.ask，在界面里正常，在 HTTP 下可能因为序列化、阻塞、异常类型而坏掉 ——
而这些坏法在本地打开页面时完全看不出来。

六件事必须成立：
  1. /healthz 报出索引规模和缓存指纹（四个维度）
  2. /chat 返回结构化答案 + 工具摘要（只给第一行，不给全文）
  3. 上游故障 → 503 + 人话，而不是 500 + traceback
  4. /chat/stream 真的按段推 token，且事件和正文分开
  5. 流式里上游故障 → 推一个 error 事件后正常收尾，不把连接吊死
  6. /eval/retrieval 返回门禁结论（零成本，跑真 BM25）

跑法：
    .venv\\Scripts\\python.exe eval\\test_api.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import rag  # noqa: E402

rag.TOOLS = "0"
rag._CACHE.clear()

FAILS = []


def check(name, cond, detail=""):
    print(f"{'✓' if cond else '✗'} {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _ok_info(answer="订单处于运输中。"):
    return {
        "answer": answer,
        "hits": [],
        "mode": "hybrid",
        "tools": [
            {
                "name": "query_order",
                "args": '{"order_id":"ORD20261115002"}',
                "result": "订单号：ORD20261115002\n状态：运输中\n签收人：张先生 138****0000",
                "status": "ok",
                "order_id": "ORD20261115002",
            }
        ],
        "t_retrieval": 0.021,
        "first_at": 0.63,
        "t_total": 1.52,
        "cost": 0.00796,
        "inp": 3326,
        "out": 116,
        "degraded": None,
        "fault": None,
    }


def fake_ask_ok(question, show_retrieval=False, on_token=None, on_event=None):
    if on_event:
        on_event({"type": "thinking"})
        on_event(
            {
                "type": "tool",
                "name": "query_order",
                "args": {"order_id": "ORD20261115002"},
                "status": "ok",
                "preview": "订单号：ORD20261115002",
            }
        )
    for piece in ("订单", "处于", "运输中。"):
        if on_token:
            on_token(piece)
    return _ok_info()


def fake_ask_boom(question, show_retrieval=False, on_token=None, on_event=None):
    raise PermissionError("Error code: 403 - Free quota exhausted")


def parse_sse(text):
    """把 SSE 文本拆成事件列表。空行是分隔符，data: 后面才是负载。"""
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                import json

                out.append(json.loads(line[len("data: "):]))
    return out


client = TestClient(api.app)

# ---------- 1. /healthz ----------
h = client.get("/healthz")
check("/healthz 200", h.status_code == 200, str(h.status_code))
body = h.json()
check("报了片段数", body.get("chunks", 0) > 0, str(body.get("chunks")))
check("指纹四维齐全", set(body.get("fingerprint", {})) == {"model", "hybrid", "rerank", "chunks"})

# ---------- 2. /chat ----------
rag.ask = fake_ask_ok
r = client.post("/chat", json={"question": "查下 ORD20261115002 到哪了"})
check("/chat 200", r.status_code == 200, str(r.status_code))
data = r.json()
check("答案原样返回", data["answer"] == "订单处于运输中。", data.get("answer"))
check("指标带回来了", data["t_total"] == 1.52 and abs(data["cost"] - 0.00796) < 1e-9)
check("工具轨迹有 1 条", len(data["tools"]) == 1, str(len(data["tools"])))
# 关键：只给第一行。给全文就是把签收人手机号散到调用方去了。
check("工具结果只给摘要", "138****0000" not in data["tools"][0]["preview"], data["tools"][0]["preview"])
check("摘要就是第一行", data["tools"][0]["preview"].startswith("订单号：ORD20261115002"))

# ---------- 3. 上游故障 → 503 + 人话 ----------
rag.ask = fake_ask_boom
r = client.post("/chat", json={"question": "满多少钱包邮？"})
check("上游故障 503", r.status_code == 503, str(r.status_code))
detail = r.json().get("detail", {})
check("错误翻成人话", "额度" in str(detail.get("message", "")), str(detail)[:120])
check("带上异常类型便于归因", detail.get("error_type") == "PermissionError")

# ---------- 4. 流式：token 分段 + 事件分离 ----------
rag.ask = fake_ask_ok
with client.stream("POST", "/chat/stream", json={"question": "查下订单"}) as resp:
    check("流式 200", resp.status_code == 200, str(resp.status_code))
    check("content-type 是 SSE", "text/event-stream" in resp.headers.get("content-type", ""))
    text = "".join(resp.iter_text())

events = parse_sse(text)
kinds = [e["type"] for e in events]
check("有 start", kinds[:1] == ["start"], str(kinds[:3]))
check("token 分了多段", kinds.count("token") >= 3, str(kinds.count("token")))
check("工具事件没混进正文", any(e["type"] == "event" and e["event"].get("type") == "tool" for e in events))
check("最后是 done", kinds[-1] == "done", str(kinds[-1]))
joined = "".join(e["text"] for e in events if e["type"] == "token")
check("拼回来等于完整答案", joined == "订单处于运输中。", joined)

# ---------- 4b. 降级路径：整段返回，没有 token 回调 ----------


def fake_ask_canned(question, show_retrieval=False, on_token=None, on_event=None):
    """模拟真实降级：预设话术整段返回，一次 on_token 都不调。

    这是线上 403 时的真实行为 —— 不补这一段，流式客户端会一直空着，
    直到 done 才突然刷出全文，用户看到的是「卡了 2 秒突然出字」。
    """
    info = _ok_info("单笔实付满 99 元包邮（偏远地区除外）。")
    info["mode"] = "degraded"
    info["degraded"] = "canned"
    info["fault"] = "PermissionDeniedError: Error code: 403"
    info["tools"] = []
    info["cost"] = 0.0
    return info


rag.ask = fake_ask_canned
with client.stream("POST", "/chat/stream", json={"question": "满多少钱包邮？"}) as resp:
    text = "".join(resp.iter_text())
events = parse_sse(text)
kinds = [e["type"] for e in events]
check("降级也推了正文", kinds.count("token") >= 1, str(kinds))
check(
    "降级正文等于整段答案",
    "".join(e["text"] for e in events if e["type"] == "token")
    == "单笔实付满 99 元包邮（偏远地区除外）。",
)
check("降级仍以 done 收尾", kinds[-1] == "done", str(kinds[-1]))
check("done 里标明降级", events[-1]["info"].get("mode") == "degraded")

# ---------- 5. 流式里故障：推 error 后收尾 ----------
rag.ask = fake_ask_boom
with client.stream("POST", "/chat/stream", json={"question": "x"}) as resp:
    text = "".join(resp.iter_text())
events = parse_sse(text)
check("流式故障推 error 事件", any(e["type"] == "error" for e in events), str([e["type"] for e in events]))
check("流式故障后不发 done", not any(e["type"] == "done" for e in events))
check("流式错误也有人话", "额度" in str([e for e in events if e["type"] == "error"][0]["message"]))

# ---------- 6. 空问题 / 门禁 ----------
r = client.post("/chat", json={"question": ""})
check("空问题 422", r.status_code == 422, str(r.status_code))

g = client.get("/eval/retrieval")
check("/eval/retrieval 200", g.status_code == 200, str(g.status_code))
gb = g.json()
check("门禁给结论", isinstance(gb.get("ok"), bool))
check("基线读的是 quality.py", gb.get("baseline", {}).get("min_pass") == 74, str(gb.get("baseline")))
check("四项检查都在", len(gb.get("checks", [])) == 4, str(len(gb.get("checks", []))))

print()
if FAILS:
    print(f"失败 {len(FAILS)} 项：{FAILS}")
    sys.exit(1)
print("全部通过")
