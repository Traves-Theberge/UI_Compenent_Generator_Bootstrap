import os
import json
from openai import OpenAI
from openai import AsyncOpenAI
from termcolor import colored
import time
import asyncio
from pydantic import BaseModel
from typing import Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the OpenAIAssistant class
class OpenAIAssistant:
    def __init__(self,
                 name="OpenAI Assistant",
                 api_key=None,
                 max_history_words=10000,
                 max_words_per_message=None,
                 json_mode=False,
                 stream=True,
                 use_async=False,
                 max_retry=10,
                 model="gpt-4o",
                 should_print_init=True,
                 print_color="green"
                 ):
        
        # Initialize model name
        self.model = model
        # Initialize assistant name
        self.name = name
        # Initialize API key or get it from environment variables
        self.api_key = api_key or self._get_api_key()
        # Initialize history list
        self.history = []
        # Initialize maximum history words
        self.max_history_words = max_history_words
        # Initialize maximum words per message
        self.max_words_per_message = max_words_per_message
        # Initialize JSON mode
        self.json_mode = json_mode
        # Initialize stream mode
        self.stream = stream
        # Initialize async mode
        self.use_async = use_async
        # Initialize maximum retry attempts
        self.max_retry = max_retry
        # Initialize print color
        self.print_color = print_color
        # Initialize system message
        self.system_message = "You are a helpful assistant."
        # Append JSON mode specific message to system message if JSON mode is enabled
        if self.json_mode:
            self.system_message += " Please return your response in JSON unless user has specified a system message."

        # Initialize the client based on async mode
        self._initialize_client()

        # Print initialization message if should_print_init is True
        if should_print_init:
            print(colored(f"{self.name} initialized with model={self.model}, json_mode={json_mode}, stream={stream}, use_async={use_async}, max_history_words={max_history_words}, max_words_per_message={max_words_per_message}", "red"))

    # Method to get API key from environment variables
    def _get_api_key(self):
        return os.getenv("OPENAI_API_KEY")

    # Method to initialize the client based on async mode
    def _initialize_client(self):
        if self.use_async:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = OpenAI(api_key=self.api_key)

    # Method to set system message
    def set_system_message(self, message=None):
        self.system_message = message or "You are a helpful assistant."
        if self.json_mode and "json" not in message.lower():
            self.system_message += " Please return your response in JSON unless user has specified a system message."

    # Async version of set_system_message
    async def set_system_message_async(self, message=None):
        self.set_system_message(message)

    # Method to add a message to the history
    def add_message(self, role, content):
        if role == "user" and self.max_words_per_message:
            content += f" please use {self.max_words_per_message} words or less"
        self.history.append({"role": role, "content": str(content)})

    # Async version of add_message
    async def add_message_async(self, role, content):
        self.add_message(role, content)

    # Method to print the length of the history
    def print_history_length(self):
        history_length = sum(len(str(message["content"]).split()) for message in self.history)
        print(f"\nCurrent history length is {history_length} words")

    # Async version of print_history_length
    async def print_history_length_async(self):
        self.print_history_length()

    # Method to clear the history
    def clear_history(self):
        self.history.clear()

    # Async version of clear_history
    async def clear_history_async(self):
        self.clear_history()

    # Method to chat with the assistant
    def chat(self, user_input, response_model: Optional[BaseModel] = None, **kwargs):
        self.add_message("user", user_input)
        return self.get_response(response_model=response_model, **kwargs)

    # Async version of chat
    async def chat_async(self, user_input, response_model: Optional[BaseModel] = None, **kwargs):
        await self.add_message_async("user", user_input)
        return await self.get_response_async(response_model=response_model, **kwargs)

    # Method to trim the history based on maximum history words
    def trim_history(self):
        words_count = sum(len(str(message["content"]).split()) for message in self.history if message["role"] != "system")
        while words_count > self.max_history_words and len(self.history) > 1:
            words_count -= len(self.history[0]["content"].split())
            self.history.pop(0)

    # Async version of trim_history
    async def trim_history_async(self):
        self.trim_history()

    # Method to get a response from the assistant
    def get_response(self, color=None, should_print=True, response_model: Optional[BaseModel] = None, **kwargs):
        if color is None:
            color = self.print_color
        
        max_tokens = kwargs.pop('max_tokens', 4000)

        retries = 0
        while retries < self.max_retry:
            try:
                if response_model:
                    response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        max_tokens=max_tokens,
                        response_format=response_model,
                        **kwargs
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        stream=self.stream,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"} if self.json_mode else None,
                        **kwargs
                    )
                
                if self.stream and not response_model:
                    assistant_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                        else:
                            content = None
                        
                        if content:
                            if should_print:
                                print(colored(content, color), end="", flush=True)
                            assistant_response += content
                    print()
                else:
                    assistant_response = response.choices[0].message.content

                if self.json_mode:
                    assistant_response = json.loads(assistant_response)

                if response_model:
                    assistant_response = response.choices[0].message.parsed

                self.add_message("assistant", str(assistant_response))
                self.trim_history()
                return assistant_response
            except Exception as e:
                print("Error:", e)
                retries += 1
                time.sleep(1)
        raise Exception("Max retries reached")

    # Async version of get_response
    async def get_response_async(self, color=None, should_print=True, response_model: Optional[BaseModel] = None, **kwargs):
        if color is None:
            color = self.print_color
        
        max_tokens = kwargs.pop('max_tokens', 4000)
        
        retries = 0
        while retries < self.max_retry:
            try:
                if response_model:
                    response = await self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        max_tokens=max_tokens,
                        response_format=response_model,
                        **kwargs
                    )
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        stream=self.stream,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"} if self.json_mode else None,
                        **kwargs
                    )

                if self.stream and not response_model:
                    assistant_response = ""
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                        else:
                            content = None
                        
                        if content:
                            if should_print:
                                print(colored(content, color), end="", flush=True)
                            assistant_response += content
                    print()
                else:
                    assistant_response = response.choices[0].message.content

                if self.json_mode:
                    assistant_response = json.loads(assistant_response)

                if response_model:
                    assistant_response = response.choices[0].message.parsed

                await self.add_message_async("assistant", str(assistant_response))
                await self.trim_history_async()
                return assistant_response
            except Exception as e:
                print("Error:", e)
                retries += 1
                await asyncio.sleep(1)
