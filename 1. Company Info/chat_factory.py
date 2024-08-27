


def get_OpenAIChat():
    # from langchain.chat_models import ChatOpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")


def get_Gemini():
    """
    Returns a Gemini model for chat from VertexAI.
    
    Returns:
        A `ChatVertexAI` model.
    """
    from langchain_google_vertexai import ChatVertexAI
    model = ChatVertexAI(model="gemini-1.5-flash")
    return model


def get_Anthropic():
    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    # model = ChatAnthropic(model="claude-3-haiku-20240307")
    return model