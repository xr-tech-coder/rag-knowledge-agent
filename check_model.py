"""换模型后的四项自检。只读：不发写请求、不改任何文件、不落盘 key。

为什么要单独一个脚本，而不是直接重跑 100 题：
换模型有三个东西会悄悄变 —— 默认是否思考、stream_options 是否被接受、
tool_calls 的分片粒度。这三个只要有一个变了，评测分数就会动，
但分不清是模型变差还是代码不适配。先花两分钟把它们单独验掉，
再看评测分数，才能把"模型的问题"和"代码的问题"分开。

用法（key 只在这个窗口里存在，不写进任何文件）：
    check_model.bat
或
    $env:API_KEY="sk-..."
    .venv\\Scripts\\python.exe check_model.py
"""

import os
import sys
import time

from openai import OpenAI, BadRequestError

API_KEY = os.getenv("API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")
TEMPERATURE = 0.1

TOOLS = [{
    "type": "function",
    "function": {
        "name": "query_order",
        "description": "按订单号查询订单状态。用户没有提供订单号时不要调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号，形如 ORD20261112001"},
            },
            "required": ["order_id"],
        },
    },
}]


def mask(k):
    return k[:6] + "..." + k[-4:] if len(k) > 12 else "(太短，可能没设对)"


def head(title):
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def sec1_basic(client):
    """连通性 + 默认是否思考 + 首字耗时。"""
    head("[1] 连通性 / 默认是否思考")
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用一句话说明退货期限是几天。"}],
        stream=True,
        temperature=TEMPERATURE,
        stream_options={"include_usage": True},
    )

    first_content = None
    first_reason = None
    n_reason_chunks = 0
    pieces = []
    usage = None

    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not getattr(chunk, "choices", None):
            continue
        d = chunk.choices[0].delta

        r = getattr(d, "reasoning_content", None)
        if r:
            n_reason_chunks += 1
            if first_reason is None:
                first_reason = time.perf_counter() - t0
        c = getattr(d, "content", None)
        if c:
            if first_content is None:
                first_content = time.perf_counter() - t0
            pieces.append(c)

    answer = "".join(pieces).strip()
    print(f"  首字（正文） : {first_content:.2f}s" if first_content else "  首字（正文） : 没收到正文")
    if first_reason is not None:
        print(f"  首个思考片   : {first_reason:.2f}s  （共 {n_reason_chunks} 片）")
        print("  >>> 结论：这个模型默认【开】思考。")
        print("      界面上会表现为「卡一下才出字」；run.bat 里 THINK=0 才能关掉。")
    else:
        print("  首个思考片   : 无")
        print("  >>> 结论：这个模型默认【不开】思考（或思考内容不下发）。")
    print(f"  正文长度     : {len(answer)} 字")
    print(f"  正文         : {answer[:120]}")
    if usage:
        print(f"  usage        : prompt {usage.prompt_tokens} / completion {usage.completion_tokens}")
    else:
        print("  usage        : 没返回（第 2 节会确认原因）")
    return first_content, first_reason


def sec2_stream_options(client):
    """stream_options 是否被接受。第 1 节没报 400 就说明接受，这里显式确认 usage。"""
    head("[2] stream_options / usage 统计")
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "回答一个字：好"}],
            stream=True,
            temperature=TEMPERATURE,
            stream_options={"include_usage": True},
        )
        usage = None
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
        if usage:
            print(f"  >>> 接受 stream_options，并返回了 usage（completion {usage.completion_tokens}）。")
            print("      界面的累计花费可以正常统计。")
        else:
            print("  >>> 请求没报错，但没拿到 usage。")
            print("      影响：界面上的花费会一直是 0，其余功能正常。")
    except BadRequestError as e:
        print(f"  >>> 被拒绝（400）：{str(e)[:200]}")
        print("      rag.py 里会退化成最简流式，能跑但拿不到精确 token 数。")


def sec3_tool_calls(client):
    """工具调用的流式分片粒度。这是换模型最容易静默坏掉的地方。"""
    head("[3] 工具调用 / 流式分片")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "帮我查一下订单 ORD20261112001 现在什么状态。"}],
        stream=True,
        temperature=TEMPERATURE,
        tools=TOOLS,
    )

    slots = {}
    chunk_count = {}
    content_pieces = []

    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        d = chunk.choices[0].delta
        if getattr(d, "content", None):
            content_pieces.append(d.content)
        for tc in (getattr(d, "tool_calls", None) or []):
            idx = tc.index if getattr(tc, "index", None) is not None else 0
            slot = slots.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            chunk_count[idx] = chunk_count.get(idx, 0) + 1
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["arguments"] += fn.arguments

    if not slots:
        print("  >>> 这一题没触发工具调用。")
        print("      可能是模型判断不需要查（也可能不支持 tools）。正文：")
        print(f"      {''.join(content_pieces)[:150]}")
        return

    for idx, s in slots.items():
        print(f"  工具 #{idx}：{s['name']}  收到 {chunk_count[idx]} 个分片")
        print(f"    arguments 原文 : {s['arguments']}")
        try:
            import json
            args = json.loads(s["arguments"])
            print(f"    拼接后能解析    : 是 -> {args}")
            if chunk_count[idx] == 1:
                print("    >>> 一次性给全，rag.py 的按 index 累加逻辑照样能用（只加了一次）。")
            else:
                print(f"    >>> 分 {chunk_count[idx]} 片到达，和 plus 行为一致，拼接逻辑无需改。")
        except Exception as e:
            print(f"    拼接后能解析    : 否 -> {e}")
            print("    >>> 这是真问题：按 index 累加后仍不是合法 JSON，拼接逻辑要跟着改。")


def sec4_enable_thinking(client):
    """enable_thinking=False 这个参数当前模型认不认。"""
    head("[4] enable_thinking=False 是否被接受")
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "回答一个字：好"}],
            stream=True,
            temperature=TEMPERATURE,
            extra_body={"enable_thinking": False},
        )
        n = 0
        for _ in stream:
            n += 1
        print(f"  >>> 接受（收到 {n} 个 chunk）。THINK=0 这条路能走通。")
    except BadRequestError as e:
        print(f"  >>> 被拒绝（400）：{str(e)[:200]}")
        print("      影响：rag.py 每次都会先失败一次再退化重试，多花一次往返。")
        print("      这不是错误，但如果模型本来就不思考，这句可以删掉。")


def main():
    if not API_KEY:
        print("[x] 没有 API_KEY。先执行 $env:API_KEY=\"sk-...\"，或用 check_model.bat。")
        sys.exit(1)

    print(f"model    = {MODEL}")
    print(f"base_url = {BASE_URL}")
    print(f"api_key  = {mask(API_KEY)}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for fn in (sec1_basic, sec2_stream_options, sec3_tool_calls, sec4_enable_thinking):
        try:
            fn(client)
        except Exception as e:
            print(f"  [x] 这一节失败：{type(e).__name__}: {str(e)[:300]}")

    head("四节跑完。把上面整段输出贴回来即可。")


if __name__ == "__main__":
    main()
