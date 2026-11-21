"""HTTP 服务层：把 rag.ask() 暴露成 REST + SSE。

为什么单独一层，而不是让系统直接调 Streamlit
----------------------------------------
界面是给人点的，API 是给系统调的。两者对三件事的期望不同：
错误（界面要一段人话，API 要一个状态码）、并发（界面一个用户，API 可能十个）、
返回（界面要排版，API 要结构）。混在一起的结果是两边都别扭。

这一层刻意很薄：没有检索、没有 prompt、没有计费逻辑，只是把 rag.ask 的
返回值和事件翻译成 HTTP。薄的好处是改 RAG 的时候不需要动它。

端点
----
GET  /healthz           存活 + 索引规模 + 生效配置（含缓存指纹）
POST /chat              完整答案 + 指标，非流式
POST /chat/stream       SSE：token 逐段推，工具事件单独推
GET  /eval/retrieval    跑检索层门禁，返回统计与是否达标（零 API 成本）

跑法
----
    .venv\\Scripts\\python.exe -m uvicorn api:app --port 8010
    curl -s localhost:8010/healthz | python -m json.tool

一个并发上的限制，先说清楚
------------------------
检索档位和工具开关是 rag 模块级的全局变量（界面侧边栏直接改它们）。
所以这一层**不接受按请求改档位**：十个请求各带一个档位，会互相串味，
而且出事时无法归因。要换档位请改服务端环境变量并重启。
"""

import contextlib
import io
import json
import queue
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import quality
import rag
import retrieval as R
from errors import friendly_error


@contextlib.asynccontextmanager
async def lifespan(app):
    # 启动时建一次索引。失败不阻断启动 —— /healthz 会暴露出来是什么状态，
    # 比「进程起不来、日志里一行 traceback」好排查。
    try:
        rag.ensure_index()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 启动时建索引失败：{type(e).__name__}: {e}")
    yield


app = FastAPI(
    title="电商售后问答 Agent",
    description="可溯源 RAG + 工具调用的 HTTP 接口。答案带依据，指标带成本。",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")


class ToolTrace(BaseModel):
    name: str
    status: str = ""
    order_id: str = ""
    preview: str = ""


class ChatResponse(BaseModel):
    answer: str
    mode: str = ""
    degraded: Optional[str] = None
    fault: Optional[str] = None
    cost: float = 0.0
    t_retrieval: float = 0.0
    first_at: float = 0.0
    t_total: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tools: list[ToolTrace] = []


def _public(info):
    """rag.ask 的返回 → 对外结构。

    工具结果只给第一行摘要，不给全文：完整返回里可能有订单人、地址、手机号这类
    字段。给摘要够用来判断「它到底查了没」，给全文就是把个人信息散到调用方去了。
    """
    tools = []
    for t in info.get("tools") or []:
        name = t.get("name", "")
        preview = ""
        if t.get("result"):
            preview = t["result"].splitlines()[0][:120]
        elif t.get("preview"):
            preview = t["preview"][:120]
        tools.append(
            ToolTrace(
                name=name,
                status=t.get("status", "") or "",
                order_id=t.get("order_id", "") or "",
                preview=preview,
            )
        )

    return ChatResponse(
        answer=info.get("answer", ""),
        mode=info.get("mode", "") or "",
        degraded=info.get("degraded"),
        fault=info.get("fault"),
        cost=round(float(info.get("cost") or 0), 6),
        t_retrieval=round(float(info.get("t_retrieval") or 0), 3),
        first_at=round(float(info.get("first_at") or 0), 3),
        t_total=round(float(info.get("t_total") or 0), 3),
        tokens_in=int(info.get("inp") or 0),
        tokens_out=int(info.get("out") or 0),
        tools=tools,
    )


def _sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/healthz")
def healthz():
    """给探活用。索引规模也报出来 —— 片段数变了，答案就该重新验一遍。"""
    n_chunks, n_docs = rag.ensure_index()
    return {
        "ok": True,
        "docs": n_docs,
        "chunks": n_chunks,
        "model": rag.MODEL,
        "hybrid": rag.HYBRID == "1",
        "rerank": rag.RERANK == "1",
        "tools": rag.TOOLS == "1",
        # 和缓存 key 用同一套维度：这四个值任意一个变了，旧答案就不该再被返回
        "fingerprint": {
            "model": rag.MODEL,
            "hybrid": rag.HYBRID,
            "rerank": rag.RERANK,
            "chunks": n_chunks,
        },
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """非流式。适合脚本、评测、以及不需要逐字看的调用方。"""
    try:
        info = rag.ask(req.question, show_retrieval=False, on_token=lambda s: None)
    except Exception as e:  # noqa: BLE001
        # 统一 503：上游（模型/额度/网络）挂了，客户端的正确反应都是「稍后重试」。
        # 用 500 会让人以为是我们的 bug，跑去查代码。
        raise HTTPException(
            status_code=503,
            detail={"message": friendly_error(e), "error_type": type(e).__name__},
        ) from e
    return _public(info)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式：token 一段段出来，工具调用事件单独一行。

    实现上有个必须说清的点：rag.ask 是阻塞的，on_token 是同步回调，
    直接在生成器里 yield 拿不到它。所以起一个线程跑 ask，回调把事件塞进队列，
    这里从队列往外吐。少了这层，就只能等全文生成完一次性返回 —— 那就不叫流式了。
    """

    def event_stream():
        q = queue.Queue()
        box = {}

        def on_token(text):
            q.put(("token", text))

        def on_event(ev):
            q.put(("event", ev))

        def work():
            try:
                box["info"] = rag.ask(
                    req.question, show_retrieval=False, on_token=on_token, on_event=on_event
                )
            except Exception as e:  # noqa: BLE001
                box["err"] = e
            finally:
                q.put(None)  # 结束哨兵：异常也要放，否则客户端一直挂着

        threading.Thread(target=work, daemon=True).start()

        yield _sse({"type": "start"})

        n_token = 0
        while True:
            item = q.get()
            if item is None:
                break
            kind, payload = item
            if kind == "token":
                n_token += 1
                yield _sse({"type": "token", "text": payload})
            else:
                yield _sse({"type": "event", "event": payload})

        # 降级路径没有 token 事件 —— 预设话术是整段返回的，不经过流式回调。
        # 不补这一段，流式客户端会一直空着，直到 done 才突然刷出全文，
        # 看起来像卡死。补成单个 token 不算"假装流式"：它本来就是一次给的。
        if "err" not in box and n_token == 0:
            answer = (box.get("info") or {}).get("answer") or ""
            if answer:
                yield _sse({"type": "token", "text": answer})

        if "err" in box:
            e = box["err"]
            yield _sse(
                {"type": "error", "message": friendly_error(e), "error_type": type(e).__name__}
            )
        else:
            yield _sse({"type": "done", "info": _public(box["info"]).model_dump()})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/eval/retrieval")
def eval_retrieval():
    """跑一次检索层门禁。零 API 成本，纯本地 BM25，约 2 秒。

    run_golden 会打印 100 行，这里重定向掉 —— 接口返回结构化数据，
    不要把控制台输出当接口输出。
    """
    rag.ensure_index()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stats = R.run_golden(rag.BM25, rag.CHUNKS)
    verdict = quality.evaluate(stats)
    return {
        "ok": verdict["ok"],
        "stats": stats,
        "checks": verdict["checks"],
        "failed": verdict["failed"],
        "baseline": quality.BASELINE,
    }
