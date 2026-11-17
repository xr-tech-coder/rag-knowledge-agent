"""
Day 1：打通一次模型调用，并量出三件事 —— token 数、花了多少钱、首字多快。

跑之前（二选一设置 key）：
    PowerShell:  $env:API_KEY="sk-你的key"
    cmd:         set API_KEY=sk-你的key

然后：
    python main.py

没有 key 也能跑，会告诉你该怎么设，不会崩。
"""

import os
import time

# ============ 配置（改这里就够了）============
API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
# 默认指向公开接口；换服务商就设环境变量 BASE_URL，不要把私有地址写进代码
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

# 价格按你实际用的模型改，单位：元 / 百万 token
# 不确定就去官网查一次，写死在这里，以后算成本都用它
INPUT_PRICE_PER_M = 2.0
OUTPUT_PRICE_PER_M = 8.0

QUESTION = "用三句话解释什么是 RAG，给一个电商售后场景的例子。"


def cost(input_tokens: int, output_tokens: int) -> float:
    """一次调用花了多少钱（元）"""
    return (
        input_tokens / 1e6 * INPUT_PRICE_PER_M
        + output_tokens / 1e6 * OUTPUT_PRICE_PER_M
    )


def main() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("缺少依赖，先执行：")
        print("  pip install -r requirements.txt")
        print(
            "（如果没建虚拟环境，先跑 python -m venv .venv && .venv\\Scripts\\activate）"
        )
        return

    if not API_KEY:
        print("没有找到 API_KEY，先设一下环境变量：")
        print('  PowerShell:  $env:API_KEY="sk-你的key"')
        print("  cmd:         set API_KEY=sk-你的key")
        print("设置完重开一个终端，再跑 python main.py")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "user", "content": QUESTION}]

    # 关思考开关：默认开着。想关就设 THINK=0（Qwen3 支持 enable_thinking=False）
    no_think = os.getenv("THINK", "1").strip().lower() in ("0", "false", "no", "off")
    extra = {"extra_body": {"enable_thinking": False}} if no_think else {}
    print(f"model={MODEL} | thinking={'关闭' if no_think else '开启(默认)'}")

    # ---------- 1) 非流式：等全部生成完才返回 ----------
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=MODEL, messages=messages, **extra)
    t_total = time.perf_counter() - t0

    msg = resp.choices[0].message
    answer = msg.content or ""
    reason = getattr(msg, "reasoning_content", None) or ""
    usage = resp.usage
    inp = usage.prompt_tokens if usage else 0
    out = usage.completion_tokens if usage else 0

    print("=" * 56)
    print("【非流式】回答：")
    print(answer)
    print("-" * 56)
    print(
        f"总耗时 {t_total:.2f}s | 输入 {inp} tokens | 输出 {out} tokens "
        f"| 花费 ￥{cost(inp, out):.6f}"
    )
    if out > len(answer) * 2 + 20:
        print(
            f"⚠ 输出 {out} token 但正文只有 {len(answer)} 字 —— 差额是思考 token，"
            f"照样计费"
        )
    if reason:
        print(f"（本次非流式还回传了思考内容 {len(reason)} 字）")

    # ---------- 2) 流式：分别记「思考首段」和「正文首字」 ----------
    # 推理模型的正文前会先推 reasoning_content，那时 delta.content 是空的。
    # 只认 content 会把"思考结束"当成首字时延，数据就会自相矛盾。
    t0 = time.perf_counter()
    first_think = None
    first_text = None
    think_chars = 0
    pieces = []
    usage2 = None
    announced = False

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},  # 让服务端在最后返回用量
            **extra,
        )
    except Exception:
        # 部分服务商不支持 stream_options，退化成普通流式（此时拿不到精确 token 数）
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, stream=True, **extra
        )

    print("\n【流式】")
    for chunk in stream:
        now = time.perf_counter() - t0
        if getattr(chunk, "usage", None):
            usage2 = chunk.usage
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            if first_think is None:
                first_think = now
            think_chars += len(rc)
            if first_text is None and not announced:
                announced = True
                print("  …思考中（这段是模型的思考，不是给用户看的正文）",
                      end="", flush=True)
        if delta.content:
            if first_text is None:
                first_text = now
                if first_think is not None:
                    print(f"\r  （思考 {now - first_think:.2f}s 后开始出正文）")
            pieces.append(delta.content)
            print(delta.content, end="", flush=True)

    t_stream = time.perf_counter() - t0
    out2 = usage2.completion_tokens if usage2 else len("".join(pieces))
    inp2 = usage2.prompt_tokens if usage2 else inp

    ft_text = first_text if first_text is not None else t_stream
    ft_think = first_think if first_think is not None else ft_text
    tpot = (t_stream - ft_text) / max(out2 - 1, 1) if first_text else 0.0

    print("\n" + "-" * 56)
    if think_chars:
        print(
            f"思考首段 {ft_think:.2f}s | 正文首字 {ft_text:.2f}s "
            f"| 全部结束 {t_stream:.2f}s"
        )
        print(f"思考内容 {think_chars} 字 | 输出 {out2} tokens "
              f"| 花费 ￥{cost(inp2, out2):.6f}")
    else:
        print(
            f"首字 {ft_text:.2f}s | 全部结束 {t_stream:.2f}s "
            f"| 平均每 token {tpot * 1000:.0f}ms | 花费 ￥{cost(inp2, out2):.6f}"
        )

    # ---------- 3) 结论（按实测数据判断，不要背固定结论）----------
    print("=" * 56)
    print(f"非流式总耗时 {t_total:.2f}s | 流式正文首字 {ft_text:.2f}s "
          f"（变化 {ft_text - t_total:+.2f}s）")

    if think_chars:
        print(f"用户实际等 {ft_text:.2f}s 才看到第一个正文，其中 "
              f"{ft_text - ft_think:.2f}s 花在思考上。")
        print("流式救不了思考段 —— 要压它只能关思考（THINK=0）或换非推理模型。")
    elif ft_text < t_total * 0.6:
        print("结论：流式确实让『用户开始看到字』更早了，总计算量一分没少。")
    elif ft_text < t_total * 0.95:
        print("结论：流式有点效果，但首字仍偏慢 —— 服务端在攒内容（缓冲）。")
    else:
        print("结论：⚠ 这次流式没让首字变快，反而更慢。")
        print("      先跑 diag_stream.py 定位，别急着下结论。")


if __name__ == "__main__":
    main()
