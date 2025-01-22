from dotenv import load_dotenv
from langgraph.graph import MessagesState

load_dotenv()


class State(MessagesState):
    pass


if __name__ == "__main__":
    print("Hello Reflexion Agent")
