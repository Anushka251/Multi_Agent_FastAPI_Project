
"""Multi-agent workflow with FAISS-based FAQ retrieval."""

from __future__ import annotations

import ast
import operator
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from rag.retrieve import retrieve_documents


class AgentState(TypedDict, total=False):
    query: str
    route: str
    specialist_answer: str
    final_answer: str
    trace: list[str]
    retrieved_documents: list[dict]


# Safe arithmetic operators
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_math(expression: str):
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)

        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float)
        ):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            left = visit(node.left)
            right = visit(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent too large")

            return _ALLOWED_OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](
                visit(node.operand)
            )

        raise ValueError("Unsupported expression")

    return visit(tree)


def planner(state: AgentState) -> AgentState:
    query = state["query"]
    q_lower = query.lower()

    math_chars = (
        any(ch.isdigit() for ch in query)
        and any(op in query for op in ["+", "-", "*", "/", "%", "×", "÷"])
    )

    faq_keywords = [
        "order", "tracking", "track", "refund", "return",
        "account", "password", "login", "log in", "forgot",
    ]

    research_keywords = [
        "latest", "current", "research", "search",
        "source", "news",
    ]

    if math_chars:
        route = "calculator"
    elif any(keyword in q_lower for keyword in faq_keywords):
        route = "research"
    elif any(keyword in q_lower for keyword in research_keywords):
        route = "research"
    else:
        route = "writer"

    return {
        **state,
        "route": route,
        "trace": state.get("trace", [])
        + [f"planner -> {route} agent"],
    }


def calculator_agent(state: AgentState) -> AgentState:
    query = state["query"].replace("×", "*").replace("÷", "/")

    candidates = []
    current = ""

    for char in query:
        if char.isdigit() or char in ".()+-*/% ":
            current += char
        else:
            if current.strip():
                candidates.append(current.strip())
            current = ""

    if current.strip():
        candidates.append(current.strip())

    candidates = [
        candidate
        for candidate in candidates
        if any(char.isdigit() for char in candidate)
        and any(op in candidate for op in "+-*/%")
    ]

    try:
        if not candidates:
            raise ValueError("No arithmetic expression found.")

        expression = max(candidates, key=len)
        value = _safe_math(expression)
        answer = f"{expression} = {value}"

    except Exception as exc:
        answer = f"Could not safely evaluate the expression: {exc}"

    return {
        **state,
        "specialist_answer": answer,
        "trace": state.get("trace", [])
        + ["calculator agent -> calculated expression"],
    }


def research_agent(state: AgentState) -> AgentState:
    """Retrieve relevant FAQ documents from the FAISS vector store."""
    query = state["query"]

    try:
        results = retrieve_documents(query, k=2)

        # Reject weak matches rather than inventing an answer.
        relevant_results = [
            result for result in results
            if result["score"] >= 0.30
        ]

        if not relevant_results:
            answer = (
                "I could not find sufficiently relevant information "
                "in the available FAQ documents. Please contact support."
            )

            return {
                **state,
                "specialist_answer": answer,
                "retrieved_documents": [],
                "trace": state.get("trace", [])
                + ["research agent -> no sufficiently relevant FAQ found"],
            }

        context = "\n\n".join(
            result["text"] for result in relevant_results
        )

        # The answer is grounded directly in retrieved FAQ text.
        answer = (
            "Based on the available FAQ information:\n\n"
            + context
        )

        trace_message = (
            f"research agent -> FAISS retrieved "
            f"{len(relevant_results)} relevant document(s)"
        )

        return {
            **state,
            "specialist_answer": answer,
            "retrieved_documents": relevant_results,
            "trace": state.get("trace", []) + [trace_message],
        }

    except Exception as exc:
        return {
            **state,
            "specialist_answer": (
                "FAQ retrieval failed. Please check the vector store "
                f"and dependencies. Details: {exc}"
            ),
            "retrieved_documents": [],
            "trace": state.get("trace", [])
            + ["research agent -> retrieval error"],
        }


def writer_agent(state: AgentState) -> AgentState:
    query = state["query"].strip()

    answer = (
        f"You asked: {query}\n\n"
        "This question is outside the capabilities of the currently "
        "configured FAQ retrieval and calculator agents. "
        "The local Qwen generation model has not yet been integrated."
    )

    return {
        **state,
        "specialist_answer": answer,
        "trace": state.get("trace", [])
        + ["writer agent -> fallback response; Qwen not configured"],
    }


def evaluator(state: AgentState) -> AgentState:
    answer = (state.get("specialist_answer") or "").strip()

    if not answer:
        answer = "The workflow did not produce an answer."

    return {
        **state,
        "final_answer": answer,
        "trace": state.get("trace", [])
        + ["evaluator -> checked that a response exists"],
    }


def choose_agent(state: AgentState) -> str:
    return state.get("route", "writer")


builder = StateGraph(AgentState)

builder.add_node("planner", planner)
builder.add_node("research", research_agent)
builder.add_node("calculator", calculator_agent)
builder.add_node("writer", writer_agent)
builder.add_node("evaluator", evaluator)

builder.add_edge(START, "planner")

builder.add_conditional_edges(
    "planner",
    choose_agent,
    {
        "research": "research",
        "calculator": "calculator",
        "writer": "writer",
    },
)

builder.add_edge("research", "evaluator")
builder.add_edge("calculator", "evaluator")
builder.add_edge("writer", "evaluator")
builder.add_edge("evaluator", END)

graph = builder.compile()


def run_agent_workflow(query: str) -> AgentState:
    return graph.invoke({"query": query, "trace": []})
