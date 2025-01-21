import datetime

from dotenv import load_dotenv
from langchain.output_parsers import JsonOutputToolsParser, PydanticToolsParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

USE_ANTHROPIC = True

if USE_ANTHROPIC:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        max_tokens=2048,  # Adjust this number as needed (up to 4096 for Claude 3 Sonnet)
    )
else:
    llm = ChatOpenAI(model="gpt-4o-mini")

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        AIMessage(
            content="Answer the user's question above using the required format."
        ),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
    res = chain.invoke(input={"messages": [human_message]})
    print(res)
