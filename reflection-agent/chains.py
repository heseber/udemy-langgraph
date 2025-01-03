from langchain.schema import SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

USE_ANTHROPIC = True

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            " Always provide detailed recommendations including requests for length, virality, style, etc."
            " If you do not have any critique or recommendation, respond with 'FINAL: No further recommendations.' as the sole response, no other text."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."
            " Return only the tweet content, no other text."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
if USE_ANTHROPIC:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        max_tokens=2048,  # Adjust this number as needed (up to 4096 for Claude 3 Sonnet)
    )
else:
    llm = ChatOpenAI(model="gpt-4o-mini")

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
