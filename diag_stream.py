"""
流式诊断 v2：区分「思考阶段」和「正文阶段」，并尝试关掉思考。

跑法：
    双击 diag.bat   或   .venv\\Scripts\\python.exe diag_stream.py
"""
import os
import statistics
import time

API_KEY = os.getenv("API_KEY", "").strip()
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

QUESTION = "用三句话解释什么是 RAG，给一个电商售后场景的例子。"
TINY = "只回复两个字：收到"

# 各家关思考的写法不一样，逐个试，哪个生效就用哪个
THINK_OFF = [
    ("enable_thinking=False (Qwen3/vLLM)", {"enable_thinking": False}),
    ("thinking.type=disabled (GLM/网关)", {"thinking": {"type": "disabled"}}),
    ("chat_template_kwargs.enable_thinking=False",
     {"chat_template_kwargs": {"enable_thinking": False}}),
]

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("缺依赖：.venv\\Scripts\\python.exe -m pip install openai")


def banner(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


def fmt(s):
    return f"{s:.2f}s"


def call_plain(client, question, max_tokens=None, extra_body=None):
    kw = dict(model=MODEL, messages=[{"role": "user", "content": question}])
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    if extra_body:
        kw["extra_body"] = extra_body
    t0 = time.perf_counter()
    r = client.chat.completions.create(**kw)
    dt = time.perf_counter() - t0
    msg = r.choices[0].message
    u = r.usage
    rc = getattr(msg, "reasoning_content", None) or ""
    return {
        "t": dt, "text": msg.content or "", "reasoning_chars": len(rc),
        "in_tok": u.prompt_tokens if u else 0,
        "out_tok": u.completion_tokens if u else 0,
        "model": getattr(r, "model", "?"),
    }


def call_stream(client, question):
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        stream=True,
        stream_options={"include_usage": True},
    )
    events, answer, thinking = [], [], []
    usage = None
    for ch in r:
        now = time.perf_counter() - t0
        u = getattr(ch, "usage", None)
        if u:
            usage = u
        if not getattr(ch, "choices", None):
            continue
        d = ch.choices[0].delta
        rc = getattr(d, "reasoning_content", None)
        if rc:
            thinking.append(rc)
            events.append({"t": now, "kind": "think", "n": len(rc)})
        c = getattr(d, "content", None)
        if c:
            answer.append(c)
            events.append({"t": now, "kind": "text", "n": len(c)})
            print(c, end="", flush=True)
    return {
        "total": time.perf_counter() - t0, "events": events,
        "text": "".join(answer),
        "reasoning_chars": sum(len(x) for x in thinking),
        "in_tok": usage.prompt_tokens if usage else 0,
        "out_tok": usage.completion_tokens if usage else 0,
        "model": getattr(r, "model", "?"),
    }


def describe(r, label):
    ev = r["events"]
    think = [e for e in ev if e["kind"] == "think"]
    text = [e for e in ev if e["kind"] == "text"]
    print(f"\n--- {label} 时间线 ---")
    print(f"总事件 {len(ev)}：思考 {len(think)} 段 / 正文 {len(text)} 段")
    if think:
        print(f"思考：首段 {fmt(think[0]['t'])} → 末段 {fmt(think[-1]['t'])}"
              f"（跨度 {fmt(think[-1]['t'] - think[0]['t'])}，"
              f"共 {r['reasoning_chars']} 字符）")
    if text:
        print(f"正文：首字 {fmt(text[0]['t'])} → 末字 {fmt(text[-1]['t'])}"
              f"（跨度 {fmt(text[-1]['t'] - text[0]['t'])}，共 {len(r['text'])} 字符）")
    if think and text:
        after = [e for e in think if e["t"] > text[0]["t"]]
        if after:
            print(f"⚠ 正文开始后仍有 {len(after)} 段思考，最后一段在 {fmt(after[-1]['t'])}"
                  f" —— 思考和正文交错")
    if len(ev) > 1:
        gaps = [(ev[i + 1]["t"] - ev[i]["t"], i) for i in range(len(ev) - 1)]
        g, i = max(gaps)
        print(f"最长空档 {fmt(g)}：{fmt(ev[i]['t'])}（{ev[i]['kind']}）"
              f" → {fmt(ev[i + 1]['t'])}（{ev[i + 1]['kind']}）")
    if len(text) > 1:
        tg = [text[i + 1]["t"] - text[i]["t"] for i in range(len(text) - 1)]
        print(f"正文 chunk 间隔：中位数 {statistics.median(tg) * 1000:.0f}ms"
              f" | 最大 {max(tg) * 1000:.0f}ms")


def main():
    if not API_KEY:
        print('没有 API_KEY。先设：$env:API_KEY="sk-你的key"')
        return
    banner("环境")
    print(f"model    = {MODEL}")
    print(f"base_url = {BASE_URL}")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    banner("T1  极小问题 + 非流式（token 有没有花在看不见的地方）")
    a = call_plain(client, TINY)
    visible = len(a["text"].strip())
    print(f"回答 {a['text']!r}（{visible} 字）")
    print(f"耗时 {fmt(a['t'])} | 输入 {a['in_tok']} | 输出 {a['out_tok']} tokens"
          f" | 实际模型 {a['model']}")
    if a["out_tok"] > visible * 2 + 10:
        print(f"⚠ 输出 {a['out_tok']} token 只换回 {visible} 字 → 其余在思考，"
              f"约占 {(1 - visible / max(a['out_tok'], 1)) * 100:.0f}%")

    banner("T1b  max_tokens=8 是否被遵守（成本能不能限住）")
    try:
        b = call_plain(client, TINY, max_tokens=8)
        print(f"设 max_tokens=8，实际输出 {b['out_tok']} tokens → "
              f"{'遵守' if b['out_tok'] <= 8 else '❌ 没遵守，这个参数管不住成本'}")
    except Exception as e:
        print(f"调用失败：{type(e).__name__}: {str(e)[:120]}")

    banner("T2  极小问题 + 流式（思考有没有被单独推过来）")
    c = call_stream(client, TINY)
    print()
    describe(c, "T2")
    if len(c["events"]) <= 3 and len(c["text"]) < 40:
        print("注：回答只有几个字，chunk 少是正常的，不能据此判假流式")

    banner("T3  原问题 + 流式（真实场景）")
    d = call_stream(client, QUESTION)
    print()
    describe(d, "T3")
    print(f"用量：输入 {d['in_tok']} / 输出 {d['out_tok']} tokens")
    if d["out_tok"]:
        print(f"整体 ≈ {d['out_tok'] / d['total']:.1f} tokens/s")
    te = [e for e in d["events"] if e["kind"] == "text"]
    if te:
        print(f"正文 ≈ {len(d['text']) / max(d['total'] - te[0]['t'], 0.01):.1f} 字/s")

    banner("T4  尝试关掉思考（哪种写法生效）")
    base = a["out_tok"]
    working = None
    for name, body in THINK_OFF:
        try:
            r = call_plain(client, TINY, max_tokens=64, extra_body=body)
            eff = "✅ 生效" if r["out_tok"] < base * 0.5 else "❌ 无效"
            print(f"{name}: 输出 {r['out_tok']} tokens（原 {base}）{eff}  {fmt(r['t'])}")
            if r["out_tok"] < base * 0.5:
                working = name
        except Exception as e:
            print(f"{name}: 不支持（{type(e).__name__}: {str(e)[:80]}）")

    banner("诊断结论")
    if d["reasoning_chars"] > 0 or a["out_tok"] > visible * 2 + 10:
        print("① 这是【推理模型】：正文前先写思考，思考 token 照样计费。")
        if te:
            print(f"   用户等 {fmt(te[0]['t'])} 才看到第一个正文，这段时间全在思考。")
        print("   流式本身是真的（正文阶段 chunk 连续到达），不是假流式。")
    else:
        print("① 不是推理模型，输出基本等于正文。")
    if working:
        print(f"② 关掉思考可行：{working}。开关一开，首字和成本立刻降。")
    else:
        print("② 三种关思考写法都没生效，直接换非推理模型更省事。")
    print("③ 客服要 1 秒内出字 → 推理模型不适合在线问答，"
          "应挪到复杂工单分析这类异步任务。")
    print("\n把上面全部输出贴回来。")


if __name__ == "__main__":
    main()
