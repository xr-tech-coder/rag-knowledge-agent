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

import re

import streamlit as st

import rag
import retrieval as R

st.set_page_config(page_title="售后政策问答", layout="centered")

# 三种检索方式。键是界面文字，值是 (HYBRID, RERANK)。
MODES = {
    "BM25（默认）": ("0", "0"),
    "混合检索（BM25 + 向量）": ("1", "0"),
    "混合 + 重排": ("1", "1"),
}

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
        render_sources(msg)

    st.session_state.messages.append(msg)
    st.session_state.total_cost += info["cost"]
    st.session_state.total_q += 1


# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.subheader("检索设置")
    mode = st.radio(
        "检索方式",
        list(MODES),
        index=0,
        help="混合检索补召回，重排只改排序。两者默认关闭，原因见 README。",
    )
    apply_mode(mode)

    st.divider()
    n_chunks, n_docs = load_index()
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
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, use_container_width=True):
            ask_and_render(q)
            st.rerun()

for m in st.session_state.messages:
    render_message(m)

if prompt := st.chat_input("问一句售后问题，例如：运费谁出？"):
    ask_and_render(prompt)
    st.rerun()
