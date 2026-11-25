
from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# ----- Tools ----
@tool
def get_product_price(product: str) -> float:
    """Lookup the product price in the catalog"""
    print(f" >> Executing get_product_price product: {product}")
    prices = {"laptop": 1000.00, "headphones": 129.95, "keyboard": 89.00}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply the discount to the price"""
    print(f" >> Executing apply_discount price : {price}, discount_tier : {discount_tier}")
    discount_Percentages = {"bronze": 5, "silver": 7, "gold": 10}
    discount_price = discount_Percentages.get(discount_tier, 0)
    return round(price * (1 - discount_price / 100), 2)


# --- agent creation--
@traceable(name="langchain agent loop")
def run_agent(question: str):
    print(f" >> Executing run_agent")
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question : {question}")
    print("=" * 60)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
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
    result = run_agent("what is the price for laptop after applying a gold discount?")
