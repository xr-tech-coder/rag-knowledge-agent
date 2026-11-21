"""接口异常 → 人话。

界面和 HTTP 服务层共用这一份。分开写两份的后果不是说错话，是**慢慢对不上**：
线上报 403 时界面提示「去控制台充值」，API 却返回一句「internal server error」，
排查的人会以为这是两件事。

返回的是 Markdown：界面直接喂 st.error，API 塞进 JSON 的 message 字段。
两边都当纯文本渲染，不依赖任何框架类型。
"""


def friendly_error(e):
    """把接口异常翻译成人话。返回 Markdown。"""
    raw = f"{type(e).__name__}: {e}"
    low = raw.lower()

    if "401" in raw or "invalid api key" in low or "authentication" in low or "unauthorized" in low:
        return (
            "**接口鉴权失败（401）**\n\n"
            "API_KEY 没被读到，或者已经失效。\n\n"
            "本地：检查 `.streamlit/secrets.toml` 或环境变量。\n"
            "线上：检查平台 Secrets 里的 `API_KEY` 有没有多余空格、引号是不是英文。"
        )
    if "403" in raw or "quota" in low or "insufficient" in low or "balance" in low or "arrears" in low:
        return (
            "**额度用完了（403）**\n\n"
            "这不是代码问题——模型服务那边没有可用额度了。\n\n"
            "去控制台充值，或关掉「仅使用免费额度」模式。"
        )
    if "rate limit" in low or "429" in raw:
        return "**触发限流（429）**\n\n问得太快了，等几秒再试。"
    if "timeout" in low or "timed out" in low:
        return (
            "**请求超时**\n\n"
            "模型服务响应太慢。重试一次通常就好了；连续超时说明网络或服务端有问题。"
        )
    if "connection" in low or "network" in low or "resolve" in low or "ssl" in low:
        return (
            "**连不上模型服务**\n\n"
            "检查网络，以及 `BASE_URL` 是否写错（末尾 `/v1` 容易漏或写重）。"
        )
    return f"**出错了**：`{type(e).__name__}`\n\n```\n{str(e)[:300]}\n```"
