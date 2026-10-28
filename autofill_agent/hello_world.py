"""
Hello world for the new stack. It checks three things:

1. Config and API key are set.
2. gpt-6-luna answers through LangChain.
3. A LangGraph graph pauses for a human answer (interrupt), saves its state to
   SQLite, and resumes later, even from a different process.

Run:
    python -m autofill_agent.hello_world
        -> prints a thread id and pauses, waiting for an answer

    python -m autofill_agent.hello_world --resume <thread_id> --answer "Two weeks"
        -> resumes from SQLite and finishes
"""

import argparse
import sqlite3
import uuid
from typing import List, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

from .config import AGENT_MODEL, AGENT_REASONING_EFFORT, DB_PATH, OPENAI_API_KEY
from .llm import get_chat_model


class HelloState(TypedDict, total=False):
    question: str
    answer: str
    final_message: str


def check_config() -> None:
    print("== Config ==")
    print(f"  OPENAI_API_KEY set : {bool(OPENAI_API_KEY)}")
    print(f"  Agent model        : {AGENT_MODEL} (reasoning effort: {AGENT_REASONING_EFFORT})")
    print(f"  Database           : {DB_PATH}")
    print()


def check_llm() -> None:
    print("== LLM check ==")
    reply = get_chat_model().invoke(
        [HumanMessage(content="Reply with one short sentence greeting a new developer.")]
    )
    print(f"  {reply.text}")
    print()


def draft_question(state: HelloState) -> HelloState:
    """Asks the model to phrase a question for the human. Runs once, before the pause."""
    reply = get_chat_model().invoke(
        [HumanMessage(content="Write one short question asking a job candidate for their notice period.")]
    )
    return {"question": reply.text.strip()}


def ask_human(state: HelloState) -> HelloState:
    """Pauses the graph. The value passed to interrupt() is shown to the human."""
    answer = interrupt({"question": state["question"]})
    return {"answer": str(answer)}


def finish(state: HelloState) -> HelloState:
    return {"final_message": f"Recorded answer: {state['answer']}"}


def build_graph(checkpointer):
    builder = StateGraph(HelloState)
    builder.add_node("draft_question", draft_question)
    builder.add_node("ask_human", ask_human)
    builder.add_node("finish", finish)
    builder.add_edge(START, "draft_question")
    builder.add_edge("draft_question", "ask_human")
    builder.add_edge("ask_human", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)


def _open_checkpointer() -> tuple[sqlite3.Connection, SqliteSaver]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return conn, saver


def _pending_question(result: dict) -> Optional[str]:
    interrupts: List = result.get("__interrupt__", [])
    if not interrupts:
        return None
    return interrupts[0].value["question"]


def start(thread_id: str) -> None:
    conn, saver = _open_checkpointer()
    try:
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke({}, config)
        question = _pending_question(result)
        if question is None:
            print("Graph finished without pausing:", result.get("final_message"))
            return
        print("== Paused for a human answer ==")
        print(f"  Question  : {question}")
        print(f"  Thread id : {thread_id}")
        print()
        print("Resume in any new process with:")
        print(f'  python -m autofill_agent.hello_world --resume {thread_id} --answer "Two weeks"')
    finally:
        conn.close()


def resume(thread_id: str, answer: str) -> None:
    conn, saver = _open_checkpointer()
    try:
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(Command(resume=answer), config)
        print("== Resumed and finished ==")
        print(f"  {result['final_message']}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autofill agent hello world")
    parser.add_argument("--resume", metavar="THREAD_ID", help="Resume a paused run")
    parser.add_argument("--answer", help="Answer to give when resuming")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the live model call")
    args = parser.parse_args()

    check_config()
    if args.resume:
        if not args.answer:
            parser.error("--resume needs --answer")
        resume(args.resume, args.answer)
        return

    if not args.skip_llm:
        check_llm()
    start(str(uuid.uuid4()))


if __name__ == "__main__":
    main()
