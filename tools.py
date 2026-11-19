"""工具层：订单查询 + 物流查询（function calling 用）。

为什么这两个不走 RAG：
    政策是静态的（退款几天、运费谁承担），写进文档里不会变；
    订单状态和物流轨迹是**实时**的，写进文档下一秒就过期。
    所以它们是接口，不是知识。

设计上的三个决定：

1. 不做意图分类器。让模型自己决定调哪个工具 —— 用 if-else 判断「这题像不像查订单」
   是最容易坏的一层，加个「帮我看看我买的那件卫衣」就崩了。
2. 权限校验放在工具里，不放在 prompt 里。prompt 只能「劝」模型，工具能「拦」。
   查别人的订单直接返回拒绝，模型再想透露也没有数据可透露。
3. 订单不存在和订单属于别人，返回**同一句话**。
   如果分开返回「不存在」和「无权查看」，就等于提供了一个订单号枚举接口：
   能查出哪些单号真实存在。真实系统不会这么写。
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK_PATH = os.path.join(HERE, "data", "mock_orders.json")

# 演示用的「当前登录用户」。真实系统里这个来自 session，不该由调用方传。
# 放在这里而不是让模型传参：模型能传的参数，就能被 prompt 注入改掉。
CURRENT_USER = os.getenv("MOCK_USER", "u_1001")

_ORDERS = None


def _load():
    global _ORDERS
    if _ORDERS is None:
        with open(MOCK_PATH, encoding="utf-8") as f:
            _ORDERS = json.load(f)
    return _ORDERS


def _find(order_id):
    """按单号找订单，返回 (order, error)。error 为 None 表示成功。"""
    data = _load()
    key = (order_id or "").strip().upper()
    for o in data["orders"]:
        if o["order_id"].upper() == key:
            # ---------- 权限边界 ----------
            if o["owner"] != CURRENT_USER:
                return None, _DENY
            return o, None
    # 不存在与不属于当前账户返回同一句，不区分
    return None, _DENY


_DENY = (
    "该订单号不存在，或不属于当前登录账户。"
    "如果用户询问的是他人的订单，请按规则 5 回答无权查看，并请对方提供自己的订单号。"
)


def query_order(order_id):
    """查订单状态。返回给模型的是纯文本，不是 JSON —— 模型读文本更少出错。"""
    o, err = _find(order_id)
    if err:
        return err

    items = "、".join(f'{i["title"]}×{i["qty"]}' for i in o["items"])
    lines = [
        f'订单号：{o["order_id"]}',
        f'订单状态：{o["status"]}',
        f'下单时间：{o["created_at"]}',
        f'实付金额：{o["paid"]:.2f} 元',
        f"商品：{items}",
    ]
    if o["signed_at"]:
        lines.append(f'签收时间：{o["signed_at"]}')
    else:
        lines.append("签收时间：尚未签收")
    if o["carrier"]:
        lines.append(f'快递：{o["carrier"]} 运单号 {o["tracking_no"]}')
    return "\n".join(lines)


def query_logistics(order_id):
    """查物流轨迹。"""
    o, err = _find(order_id)
    if err:
        return err

    if not o["traces"]:
        return f'订单 {o["order_id"]} 当前状态为「{o["status"]}」，还没有物流轨迹（未发货）。'

    lines = [f'订单 {o["order_id"]}（{o["carrier"]} {o["tracking_no"]}）物流轨迹：']
    for i, t in enumerate(o["traces"]):
        mark = "  ← 最新" if i == len(o["traces"]) - 1 else ""
        lines.append(f'{t["time"]}  {t["desc"]}{mark}')
    lines.append(f'当前状态：{o["status"]}')
    return "\n".join(lines)


# ============ function calling 描述 ============
# description 写得越具体，模型选错的概率越低。
# 特别写明「政策类问题不要调用」—— 否则「退款几天」也会去查订单。
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": (
                "查询当前登录用户的订单状态，包括下单时间、签收时间、实付金额、商品明细。"
                "适用于「我的订单到哪了」「我这个单子签收了吗」「我买了什么」这类"
                "需要具体订单信息的问题。查询政策的通用问题（如「退款几天到账」）"
                "不需要调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，形如 ORD20260000000（ORD + 11 位数字）。用户没提供时先向用户索取，不要编造。",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logistics",
            "description": (
                "查询某个订单的物流轨迹和当前所在位置。"
                "适用于「快递到哪了」「为什么还没到」这类需要实时物流节点的问题。"
                "只问「一般几天到」这种配送时效政策时不要调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    # 示例单号必须是一个查无此单的号。原来写的是 ORD20261101001，
                    # 它在 mock 数据里真实存在 —— 用户只说「我的快递到哪了」时，
                    # 模型把示例当默认值填了进去，工具照单执行，把真实订单捞了出来。
                    # 示例本身不能是有效数据，否则它就是一个会被抄走的默认参数。
                    "order_id": {"type": "string", "description": "订单号，形如 ORD20260000000"}
                },
                "required": ["order_id"],
            },
        },
    },
]

_FUNCS = {"query_order": query_order, "query_logistics": query_logistics}


def dispatch(name, arguments):
    """执行一个工具调用。任何异常都变成文本返回给模型，不让问答挂掉。"""
    fn = _FUNCS.get(name)
    if fn is None:
        return f"未知工具：{name}"
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return f"参数不是合法 JSON，无法执行 {name}。"
    order_id = args.get("order_id", "")
    if not order_id:
        return "缺少 order_id 参数，请先向用户询问订单号。"
    return fn(order_id)


if __name__ == "__main__":
    # 自检：四种情况各跑一遍，不调用模型，零成本
    print("当前登录用户:", CURRENT_USER)
    for oid in ("ORD20261115002", "ORD20261101001", "ORD20261112004", "ORD99999999"):
        print(f"\n=== query_order({oid}) ===")
        print(query_order(oid))
    print("\n=== query_logistics(ORD20261115002) ===")
    print(query_logistics("ORD20261115002"))
    print("\n=== query_logistics(ORD20261118003) 未发货 ===")
    print(query_logistics("ORD20261118003"))
