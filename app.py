"""Streamlit 界面：把 rag.py 的问答能力放到浏览器里。

跑法：
    .venv\\Scripts\\streamlit.exe run app.py
    # 或  .venv\\Scripts\\python.exe -m streamlit run app.py

设计上的三条界线（不是风格问题，是不这么做会返工）：

1. 界面不碰检索和 prompt。所有生成都走 rag.ask() —— 流式逻辑只有一份，
   以后接工具调用（查订单/查物流）只改 rag 那一层，这个文件不用动。
2. 索引和向量用 cache_resource 缓存。Streamlit 每次交互都会重跑整个脚本，
   不缓存就等于每问一句都重建一次索引、重算一遍向量（那步要钱）。
3. 侧边栏的开关直接改 rag.HYBRID / rag.RERANK。这两个值在 retrieve() 里
   每次调用都读，所以运行中切换立即生效，不用重启。
"""

import os
import re

import streamlit as st

# 托管平台（Streamlit Cloud / Hugging Face）的密钥放在平台自己的 Secrets 里，
# 不会自动变成环境变量。必须在 import rag 之前注入 —— rag 是在模块级就读
# API_KEY / BASE_URL / MODEL 的，晚一步读到的就是默认值（DeepSeek 的地址 + 空 key），
# 表现是「界面能打开，一问就报 401」。
# 用 setdefault：本地已经设了环境变量时，平台配置不覆盖本地，避免调试时被悄悄改掉。
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass  # 本地没有 secrets.toml，这是正常情况

import rag
import retrieval as R

st.set_page_config(page_title="售后政策问答", layout="centered")

# 三种检索方式。键是界面文字，值是 (HYBRID, RERANK)。
#
# 默认是「混合」，不是 BM25。原因不是偏好，是实测：
# BM25 只认字面，「满多少钱包邮」这条的 top3 全是运费险段落和工单页，
# 含「99 元包邮」的片段一个都没进来，模型只能答「资料中未提及」；
# 「运费险最多能赔多少」更彻底，top3 里连 04-运费规则.md 都没有。
# 换成混合后两题都答对，检索只多 12ms —— 这不是取舍，是白捡的。
# 重排仍默认关：它只改排序（+850ms），该不该开由使用者按场景决定。
MODES = {
    "BM25（仅字面匹配）": ("0", "0"),
    "混合检索（BM25 + 向量）": ("1", "0"),
    "混合 + 重排": ("1", "1"),
}
DEFAULT_MODE = 1  # 界面首屏停在这一档

MODE_LABEL = {
    "bm25": "BM25",
    "hybrid": "BM25 + 向量",
    "bm25+rerank": "BM25 → 重排",
    "hybrid+rerank": "BM25 + 向量 → 重排",
}

EXAMPLES = [
    "七天无理由从哪天算？",
    "满多少钱包邮？",
    "运费险最多能赔多少？",
    "你们能给我全额退款再额外补偿 200 元吗？",
]

st.markdown(
    """
    <style>
      .block-container {max-width: 860px; padding-top: 2.2rem;}
      [data-testid="stChatMessage"] {padding: 0.2rem 0;}
      .src {
        border-left: 2px solid #d9dce1; padding: 0.35rem 0 0.35rem 0.7rem;
        margin-bottom: 0.6rem; font-size: 0.86rem; color: #4a4f57;
      }
      .src b {color: #1f2328; font-size: 0.9rem;}
      .src .meta {color: #8b919a; font-size: 0.78rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- 后端资源：整个进程只加载一次 ----------------
@st.cache_resource(show_spinner=False)
def load_index():
    """建一次索引。Streamlit 会反复重跑脚本，这里必须缓存。"""
    chunks = R.load_chunks()
    bm25 = R.build_index(chunks)
    rag.CHUNKS, rag.BM25 = chunks, bm25
    return len(chunks), len({s for s, _ in chunks})


@st.cache_resource(show_spinner=False)
def load_vectors():
    """片段向量。第一次要调 embedding 接口，之后走本地 npz 缓存。"""
    return rag.init_vectors()


@st.cache_resource(show_spinner=False)
def load_reranker():
    """重排模型。加载 20 秒左右，失败返回 False（rag 那边会降级）。"""
    return rag.init_reranker() is not None


def apply_mode(mode):
    """按界面选择设置全局开关，并初始化需要的后端。失败就降级。"""
    hybrid, rerank = MODES[mode]
    rag.HYBRID, rag.RERANK = hybrid, rerank

    if hybrid == "1":
        try:
            load_vectors()
        except Exception as e:
            rag.HYBRID = "0"
            st.warning(f"向量初始化失败，已退回纯 BM25：{e}")
    if rerank == "1":
        try:
            if not load_reranker():
                st.warning("重排模型加载失败，本次退回不重排。")
        except Exception as e:
            st.warning(f"重排不可用，已退回不重排：{e}")


# ---------------- 渲染 ----------------
def score_text(mode, score):
    """三种分数的量纲不同，显示精度也不同（RRF 只有 0.0x 量级）。"""
    if mode.endswith("+rerank"):
        return f"重排分 {score:.2f}"
    if mode == "hybrid":
        return f"融合分 {score:.4f}"
    return f"分数 {score:.2f}"


def render_tools(msg):
    """显示这次问答调了哪些工具。

    必须显示：调错工具和没调工具，在最终答案里往往看不出来
    （比如模型拿文档里的时效冒充实际物流，答案读着也通顺）。
    把调用记录摆出来，才能一眼看出"它到底查没查"。
    """
    tools = (msg.get("info") or {}).get("tools") or []
    if not tools:
        return
    for t in tools:
        with st.expander(f"🔧 调用工具 {t['name']}({t['args']})"):
            st.code(t["result"], language="text")


def render_sources(msg):
    hits = msg.get("hits") or []
    answer = msg.get("content", "")
    mode = msg.get("mode", "bm25")
    if not hits:
        return

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    with st.expander(f"引用来源（{len(hits)} 段，已引用 {sorted(cited) or '无'}）"):
        for i, (src, score, text) in enumerate(hits, 1):
            body = text.strip()
            clipped = body[:260] + ("…" if len(body) > 260 else "")
            st.markdown(
                f'<div class="src"><b>[{i}] {src}</b>'
                f'<span class="meta"> · {score_text(mode, score)}'
                f'{" · 已引用" if i in cited else ""}</span><br>{clipped}</div>',
                unsafe_allow_html=True,
            )


def render_metrics(msg):
    info = msg.get("info") or {}
    if not info:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("检索", f"{info.get('t_retrieval', 0) * 1000:.0f} ms")
    c2.metric("首字", f"{info.get('first_at', 0):.2f} s")
    c3.metric("总耗时", f"{info.get('t_total', 0):.2f} s")
    c4.metric("花费", f"￥{info.get('cost', 0):.5f}")
    st.caption(
        f"检索方式：{MODE_LABEL.get(info.get('mode', ''), info.get('mode', ''))} · "
        f"输入 {info.get('inp', 0)} tokens（含资料）/ 输出 {info.get('out', 0)}"
    )


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_metrics(msg)
            render_tools(msg)
            render_sources(msg)


def ask_and_render(question):
    """一次问答：检索 → 流式生成 → 落进会话历史。"""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        box = st.empty()
        box.markdown("_检索资料中…_")
        buf = []

        def on_token(t):
            buf.append(t)
            box.markdown("".join(buf) + "▌")

        info = rag.ask(question, show_retrieval=False, on_token=on_token)
        box.markdown(info["answer"])

        msg = {
            "role": "assistant",
            "content": info["answer"],
            "hits": info["hits"],
            "mode": info["mode"],
            "info": info,
        }
        render_metrics(msg)
        render_tools(msg)
        render_sources(msg)

    st.session_state.messages.append(msg)
    st.session_state.total_cost += info["cost"]
    st.session_state.total_q += 1


# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.subheader("检索设置")

    # 索引必须先建好，再设检索方式：混合检索是在已有片段上算向量的。
    # 顺序反了会报 "CHUNKS is not defined" 并降级成 BM25 —— 因为以前默认走 BM25、
    # 从不触发向量，这个顺序问题一直藏着，把默认档改成混合才暴露出来。
    n_chunks, n_docs = load_index()

    mode = st.radio(
        "检索方式",
        list(MODES),
        index=DEFAULT_MODE,
        help="混合检索补的是语义召回（+12ms）；重排只改排序（+850ms）。"
        "重排换更准的 top1，但对「压根没召回到」的题没用。",
    )
    apply_mode(mode)

    use_tools = st.checkbox(
        "启用工具调用",
        value=(rag.TOOLS == "1"),
        help="开启后模型可查订单和物流（模拟数据）。政策类问题仍然走检索，不会去查订单。",
    )
    rag.TOOLS = "1" if use_tools else "0"

    st.divider()
    st.caption(f"知识库：{n_docs} 份文档 / {n_chunks} 个片段")
    st.caption(f"模型：{rag.MODEL}")
    st.caption(f"接口：{rag.BASE_URL}")
    st.caption(f"温度：{rag.TEMPERATURE} · 思考：{'开' if rag.THINK == '1' else '关'}")

    st.divider()
    total_cost = st.session_state.get("total_cost", 0.0)
    total_q = st.session_state.get("total_q", 0)
    st.caption(f"本次会话：{total_q} 问 · 累计 ￥{total_cost:.4f}")
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_cost = 0.0
        st.session_state.total_q = 0
        st.rerun()

# ---------------- 主区 ----------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("total_cost", 0.0)
st.session_state.setdefault("total_q", 0)

st.title("售后政策问答")
st.caption(
    "答案只来自 data/ 下的 10 份政策文件，每句都标出处。"
    "资料里没有的问题会明确拒答，不猜。"
)

if not rag.API_KEY:
    st.error(
        "没有找到 API_KEY。先在终端设置环境变量再启动：\n\n"
        '`$env:API_KEY="sk-..."` · `$env:BASE_URL="..."` · `$env:MODEL="..."`'
    )
    st.stop()

if not st.session_state.messages:
    st.write("")
    picks = list(EXAMPLES)
    if use_tools:
        # 工具模式下列出可查的模拟订单，否则演示时不知道该问哪个单号
        import tools as _T
        picks = ["我的订单 ORD20261115002 到哪了？", "ORD20261101001 签收了吗？"] + picks
        st.caption(
            f"工具已开启，当前登录用户 `{_T.CURRENT_USER}`；"
            "可查的模拟订单：ORD20261115002（运输中）、ORD20261101001（已签收）、"
            "ORD20261118003（待发货）。ORD20261112004 属于其他用户，用来演示越权拦截。"
        )
    cols = st.columns(2)
    for i, q in enumerate(picks):
        if cols[i % 2].button(q, use_container_width=True):
            ask_and_render(q)
            st.rerun()

for m in st.session_state.messages:
    render_message(m)

if prompt := st.chat_input("问一句售后问题，例如：运费谁出？"):
    ask_and_render(prompt)
    st.rerun()
