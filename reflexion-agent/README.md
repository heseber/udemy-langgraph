# Reflexion Loop

This is a simple example of a reflexion loop using LangGraph. It is based on the Udemy course on LangGraph by Eden Marco.

> [!WARNING]
> This is an experimental project and the code may break with future updates to LangGraph and LangChain.

Changes I made:
- Updated the code to make it compatible with the latest LangGraph and LangChain versions. 
- Modified the system prompt for the first responder chain.
- Modified the system prompt for the revisor chain, also fixing the bug with the references not being shown in answers.
- Use a State class that inherits from MessagesState to store the state of the reflexion loop.
    - Added a new field for the references. This is a complete list of references found in all internet searches which may or may not be used in the final answer. This was done to demonstrate how to store additional information in the state.
    - Added a new field for the iteration number. This is used for the loop condition instead of counting the number of ToolMessages.
    - Added a new field for the maximum number of iterations.
- Run internet searches and llm calls asynchronously.
- Make it work with both OpenAI and Anthropic.
- Added several checks for the tool calls and messages.


