import os

from fastapi import FastAPI, Header, HTTPException, Query
from typing import Optional, Union, Dict, Annotated

from fastapi.responses import Response, StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

import chatbot
import summary_generator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_auth_token(auth_token: str) -> bool:
    valid_tokens = os.getenv("AUTH_TOKEN", "").split(",")
    valid_tokens = [token.strip() for token in valid_tokens]  # Remove leading and trailing spaces
    return auth_token in valid_tokens


@app.post("/chat")
def chat_message(text_prompt: str,
                 name: str = 'User',
                 instruction_name: str = Query("_", description=f"Available instruction configs: {', '.join(chatbot.instructions.keys())}"),
                 disable_history: bool = Query(False, description="Disable chat history and memory management.\n(Enabling does nothing if instruction config disables history)."),
                 x_auth_token: Annotated[str | None, Header()] = None
                 ):
    """
    Send a chat message to the chatbot.
    """
    if not validate_auth_token(x_auth_token):
        return HTTPException(status_code=401, detail="Invalid x_auth_token")

    if instruction_name not in chatbot.instructions.keys():
        return HTTPException(status_code=400, detail="Invalid instruction name. instruction config not found.")

    message = chatbot.message(text_prompt, name=name, instruction=instruction_name, disable_history=disable_history)

    return Response(content=message, media_type="text/plain")


@app.post("/chat_stream")
async def chat_message_stream(text_prompt: str,
                 name: str = 'User',
                 instruction_name: str = Query("_", description=f"Available instruction configs: {', '.join(chatbot.instructions.keys())}"),
                 disable_history: bool = Query(False, description="Disable chat history and memory management.\n(Enabling does nothing if instruction config disables history)."),
                 x_auth_token: Annotated[str | None, Header()] = None
                 ):
    """
    Send a chat message to the chatbot and stream the response.
    Because it is streamed, it will not do any answer cleanup.
    """
    if not validate_auth_token(x_auth_token):
        return HTTPException(status_code=401, detail="Invalid x_auth_token")

    if instruction_name not in chatbot.instructions.keys():
        return HTTPException(status_code=400, detail="Invalid instruction name. instruction config not found.")

    message = chatbot.message_stream(text_prompt, name=name, instruction=instruction_name, disable_history=disable_history)

    return StreamingResponse(message, media_type="text/event-stream")


@app.post("/summary")
def summary(text: str, max_length: int = 142, x_auth_token: Annotated[str | None, Header()] = None):
    """
    Generate a summary of the input text.
    """
    if not validate_auth_token(x_auth_token):
        return HTTPException(status_code=401, detail="Invalid x_auth_token")

    text = summary_generator.summarize(text, max_length=max_length)

    return Response(content=text, media_type="text/plain")


@app.post("/inject_memory")
def inject_memory(text: str, user: str = 'AI', instruction_name: str = "_", x_auth_token: Annotated[str | None, Header()] = None):
    """
    Inject a memory entry into the instruction config.

    If user == "AI", memory will be saved with bot username from the instructions.

    Returns: "SUCCESS" string if successful.
    """
    if not validate_auth_token(x_auth_token):
        return HTTPException(status_code=401, detail="Invalid x_auth_token")

    message = chatbot.inject_memory(text, name=user, instruction=instruction_name)
    if message is None:
        return HTTPException(status_code=400, detail="could not inject memory into instruction set, Most likely the instruction has save_history disabled.")

    return Response(content=message, media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
