from euriai.langchain import create_chat_model

API_KEY = None
MODEL_NAME = "gpt-4.1-nano"
TEMP = 0.7

def load_chat_engine(api_key: str = None):
    return create_chat_model(
        api_key=api_key or API_KEY,
        model=MODEL_NAME,
        temperature=TEMP
    )

def query_chat_engine(engine, prompt: str):
    reply = engine.invoke(prompt)
    return reply.content
