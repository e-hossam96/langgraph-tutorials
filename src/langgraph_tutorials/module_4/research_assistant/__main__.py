import argparse
import os
import uuid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research assistant graph.")
    parser.add_argument("--topic", type=str, default="Generative AI in E-Learning", help="Research topic.")
    parser.add_argument("--num-researchers", type=int, default=5, help="Number of researchers to spawn.")
    parser.add_argument("--num-interview-turns", type=int, default=8, help="Number of turns per interview.")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "groq", "ollama"],
        default="openai",
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name to use (defaults to the chosen provider's default model).",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=None,
        help="Max number of concurrent LLM calls (across parallel researcher branches). "
        "Defaults to unlimited; lower this for local providers like Ollama that can't handle many "
        "concurrent requests.",
    )
    return parser.parse_args()


args = parse_args()

# langgraph_tutorials.llms builds its `default_llm` singleton at import time, so the provider/model
# must be selected via env vars before `.report` (which imports it via `.base`) is imported below.
os.environ["LLM_PROVIDER"] = args.provider
if args.model_name:
    os.environ[f"{args.provider.upper()}_MODEL_NAME"] = args.model_name
if args.llm_concurrency is not None:
    os.environ["LLM_CONCURRENCY"] = str(args.llm_concurrency)

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from rich import print  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402

from langgraph_tutorials.tracers import langfuse_handler  # noqa: E402

from .report import report_graph_builder  # noqa: E402

console = Console()


async def main():
    graph = report_graph_builder.compile(checkpointer=InMemorySaver())

    with open("graph.md", "w+", encoding="utf-8") as f:
        f.write(f"""```mermaid
{graph.get_graph(xray=True).draw_mermaid(with_styles=False)}
```
""")

    thread_id = uuid.uuid4().hex
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": thread_id,
        },
    }

    async for event in graph.astream(
        input={
            "topic": args.topic,
            "num_researchers": args.num_researchers,
            "num_interview_turns": args.num_interview_turns,
        },
        config=config,
        stream_mode="updates",
    ):
        print(event)
        print("-" * 40)

    final_state = await graph.aget_state(config)
    console.print(Markdown(final_state.values["report"]))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
