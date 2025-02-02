import os

from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

# Get the LLM, either Anthropic Sonnet 3.5 or OpenAI gpt4-o-mini
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "true").lower() == "true"
if USE_ANTHROPIC:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        max_tokens=2048,  # Adjust this number as needed (up to 4096 for Claude 3 Sonnet)
        temperature=0,
    )
else:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define the system message used for both the first responder and for the revisor
system_message_template = """
    You are an expert researcher.

    1. {first_instruction}
    2. Reflect and critique your answer.
    3. {search_instruction}
"""

# Define the system messages for the first responder
system_message_first_responder = system_message_template.format(
    first_instruction="Provide a detailed ~250 word answer for the user's question.",
    search_instruction="""Recommend 1-3 specific search queries to research information and improve your answer. 
    Important: You must provide exactly 1-3 search queries, no more.""",
)

# Define the system messages for the revisor
system_message_revisor = system_message_template.format(
    first_instruction="""Revise the content with these requirements:
        1.1. Incorporate information from references using [n] citations
        1.2. Address all critique points
        1.3. Maintain a professional tone
        1.4. You must end the answer with a 'References' section listing all references used.
             Mark parts of the answer that are based on references with [n] citations.
             Use this format for the 'References' section:

             References:
             - [1] https://example.com
             - [2] https://example.com
        """,
    search_instruction="""Recommend additional search queries to research information and improve your answer.
    Important: You must provide exactly 1-3 search queries, no more.""",
)

# Define the first responder prompt template
first_responder_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_message_first_responder),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the revisor prompt template
revisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_message_revisor),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the first responder chain
first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice="AnswerQuestion",
)

# Define the revisor chain
revisor_chain = revisor_prompt_template | llm.bind_tools(
    tools=[ReviseAnswer],
    tool_choice="ReviseAnswer",
)


# Main function
if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    chain = first_responder_chain
    res = chain.invoke(input={"messages": [human_message]})
    print(res)
