import json
import os
import re
import openai
import backoff
import time
import yaml
from openai import RateLimitError, APIError, APIConnectionError, AuthenticationError
import anthropic
from google import genai
from google.genai import types
from google.generativeai.types import GenerationConfig
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI



CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurations.yaml")

_DEFAULT_GENERATION = {"max_tokens": 6000, "top_p": 1.0, "max_retries": 20}


def generation_settings():
    """Decoding and retry settings, read from configurations.yaml."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            settings = (yaml.safe_load(f) or {}).get("generation") or {}
    except FileNotFoundError:
        settings = {}
    return {**_DEFAULT_GENERATION, **{k: v for k, v in settings.items() if k in _DEFAULT_GENERATION}}


GENERATION = generation_settings()


def _max_retries():
    return GENERATION["max_retries"]


def _giveup(exc):
    """Never retry on a bad key: it will fail identically every time."""
    return isinstance(exc, AuthenticationError)


def create_client(model):
    if model.startswith("claude-"):
        client = OpenAI(
            api_key=os.environ['ANTHROPIC_API_KEY'],  # Your Anthropic API key
            base_url="https://api.anthropic.com/v1/"  # Anthropic's API endpoint
        )
        return client, model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        return anthropic.AnthropicBedrock(), client_model
    
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif model.startswith("Azure"):
        print(f"Using Azure API with model {model}.")
        # For Serverless API or Managed Compute endpoints
        client = ChatCompletionsClient(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_OPENAI_KEY"))
        )
        return client, model

    elif 'gpt' in model or model.startswith(('o1', 'o3', 'o4')):
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model in ["deepseek-chat", "deepseek-reasoner"]:
        print(f"Using DeepSeek API with {model}.")
        return OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        ), model
    elif model == "llama3.1-405b":
        print(f"Using OpenAI API with {model}.")
        return openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        ), "meta-llama/llama-3.1-405b-instruct"
    elif "gemini" in model:
        print(f"Using Google Generative AI with model {model}.")
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                      http_options={'api_version': 'v1alpha'})
        return client, model
    else:
        raise ValueError(f"Model {model} not supported.")

class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float=0) -> None:
        """Create an agent
        Args:
            model_name(str): model name
            name (str): name of this agent
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            sleep_time (float): sleep because of rate limits
        """
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.client, self.model = create_client(model_name)
        self.MAX_NUM_TOKENS = GENERATION["max_tokens"]
        self.top_p = GENERATION["top_p"]

    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, APIConnectionError),
        max_tries=_max_retries,
        giveup=_giveup,
    )
    def query(self, messages, max_tokens=None, temperature=None):
        """Query the model using appropriate API based on model type"""
        time.sleep(self.sleep_time)
        temperature = temperature if temperature is not None else self.temperature
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        self.system_message = system_message
        msg_history = [msg for msg in messages if msg["role"] != "system"]
        last_message = msg_history[-1]["content"] if msg_history else ""

        if "claude" in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_message}, *msg_history],
                temperature=temperature,
                max_tokens=max_tokens or self.MAX_NUM_TOKENS,
                n=1,
                stop=None
            )
            return response.choices[0].message.content

        
        elif "gemini" in self.model:
            gemini_contents = []
            for m in msg_history:
                gemini_contents.append({
                    "role": m["role"],
                    "parts": [{"text": m["content"]}]
                })
            response = self.client.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_message,
                    max_output_tokens=max_tokens or self.MAX_NUM_TOKENS,
                    temperature=temperature,
                    candidate_count=1,
                )
            )
            return response.text
            
        
        elif "Azure" in self.model:
            model = self.model.split("_")[-1]
            messages = [SystemMessage(content=system_message)] + [
            UserMessage(content=msg["content"]) if msg["role"] == "user"
            else AssistantMessage(content=msg["content"])
            for msg in msg_history
            ]
            response = self.client.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or self.MAX_NUM_TOKENS,
                top_p=self.top_p,
            )
            return response.choices[0].message.content

        elif self.model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12", "o3-mini-2025-01-31","o4-mini-2025-04-16"]:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_message}, *msg_history],
                max_completion_tokens=max_tokens or self.MAX_NUM_TOKENS,
                n=1,
                stop=None
            )
            return response.choices[0].message.content

        else:  # OpenAI-like APIs (including DeepSeek, OpenRouter)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_message}, *msg_history],
                temperature=temperature,
                top_p=self.top_p,
                max_tokens=max_tokens or self.MAX_NUM_TOKENS,
                n=1,
                stop=None
            )
            return response.choices[0].message.content

    def set_meta_prompt(self, meta_prompt: str):
        """Set the meta_prompt"""
        self.memory_lst = [{"role": "system", "content": meta_prompt}]

    def add_event(self, event: str):
        """Add a new event in the memory"""
        self.memory_lst.append({"role": "user", "content": event})

    def add_memory(self, memory: str):
        """Add assistant's response to memory"""
        self.memory_lst.append({"role": "assistant", "content": memory})

    def reset_memory(self):
        """Reset memory"""
        self.memory_lst = []
        self.memory_lst.append({"role": "system", "content": self.system_message})

    def ask(self, temperature: float=None):
        """Query for answer"""
        return self.query(self.memory_lst, temperature=temperature)
