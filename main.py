from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from OpenAIAssistant import OpenAIAssistant
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class HTMLComponent(BaseModel):
    html: str
    explanation: str

class AIResponse(BaseModel):
    components: List[HTMLComponent]

openai_client = OpenAIAssistant(model="gpt-4o-2024-08-06")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
def generate_components(request: Request, user_input: str = Form(...)):
    system_message = """You are an AI assistant that generates HTML components using Bootstrap classes.
    The user will provide instructions, and you should respond with appropriate HTML components.
    Only use Bootstrap classes for styling. Do not use any custom CSS or other CSS frameworks. 
    Return your response in JSON format"""
    
    openai_client.set_system_message(system_message)
    
    openai_client.add_message("user", user_input)
    response = openai_client.get_response(response_model=AIResponse)
    print(response)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "components": response.components,
        "user_input": user_input
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)