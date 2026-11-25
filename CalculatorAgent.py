from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 3
MODEL = "qwen3:1.7b"


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    print(f" >> Executing add : a {a}, b:{b}")
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    print(f" >> Executing subtract : a {a}, b:{b}")
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    print(f" >> Executing multiply : a {a}, b:{b}")
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    print(f" >> Executing divide : a {a}, b:{b}")
    return a / b


# --- agent creation--
@traceable(name="langchain Math Agent")
def run_agent(question: str):
    print(f" >> Executing run_agent")
    tools = [add, subtract, multiply, divide]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question : {question}")
    print("=" * 60)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful Math Agent"
                "You have access to a tool and do the calculations.\n\n"
                "STRICT RULES — you must follow these exactly:\n"


            )
        ),
        HumanMessage(content=question),
    ]
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f">> Iteration {iteration} -- ")
        ai_messages = llm_with_tools.invoke(messages)
        tool_calls = ai_messages.tool_calls
        if not tool_calls:
            print(f"Final answer : {ai_messages.content}")
            return ai_messages.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"Tool selected {tool_name} with args {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            return ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"Observation : {observation}")
        messages.append(ai_messages)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR :Reached MAX Iterations without answer")
    return None


if __name__ == "__main__":
    print(f" >> Hello langchain Agent")
    result = run_agent("what is the add of 1 + 2?")
