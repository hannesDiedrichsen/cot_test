import asyncio
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from google import genai
from google.genai import types
import os

@dataclass
class PromptResponse:
    stage1_prompt: str
    stage1_response: str
    stage2_prompt: str
    stage2_response: str
    total_time_ms: int
    error: Optional[str] = None

class GeminiClient:
    def __init__(self, config: Dict[str, Any]):
        # The client gets the API key from the environment variable `GEMINI_API_KEY`
        self.client = genai.Client()
        self.config = config
        self.rate_limit = config['api']['rate_limit_per_minute']
        self.timeout = config['api']['timeout_seconds']
        self.max_retries = config['api']['max_retries']
        self.thinking_budget = config['api'].get('thinking_budget', None)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_interval = 60.0 / self.rate_limit  # seconds between requests
        
        # Follow-up prompts by dataset type
        self.followup_prompts = {
            "numeric": "The final answer is: (provide only the number)",
            "multiple_choice": "The final answer is: (provide only the letter A, B, C, D, or E)",
            "binary": "The final answer is: (provide only yes or no)",
            "letters": "The final answer is: (provide only the letters)"
        }
    
    async def ask_question(self, question: str, method: str, dataset_type: str, model_name: str) -> PromptResponse:
        start_time = time.time()
        
        # Rate limiting
        await self._rate_limit()
        
        # Stage 1: Initial prompt
        if method == "zero_shot":
            stage1_prompt = question
        elif method == "zero_shot_cot":
            stage1_prompt = f"{question}\n\nLet's think step by step."
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Stage 2: Follow-up prompt
        stage2_prompt = self.followup_prompts.get(dataset_type, self.followup_prompts["numeric"])
        
        try:
            # Stage 1: Initial question
            stage1_response = await self._make_request_with_retry(stage1_prompt, model_name)
            
            # Stage 2: Follow-up with conversation context
            # Create a multi-turn conversation
            conversation = [
                genai.types.Content(role="user", parts=[genai.types.Part(text=stage1_prompt)]),
                genai.types.Content(role="model", parts=[genai.types.Part(text=stage1_response)]),
                genai.types.Content(role="user", parts=[genai.types.Part(text=stage2_prompt)])
            ]
            
            stage2_response = await self._make_request_with_retry_conversation(conversation, model_name)
            
            total_time = int((time.time() - start_time) * 1000)
            
            return PromptResponse(
                stage1_prompt=stage1_prompt,
                stage1_response=stage1_response,
                stage2_prompt=stage2_prompt,
                stage2_response=stage2_response,
                total_time_ms=total_time
            )
            
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            logging.error(f"Error in ask_question: {str(e)}")
            
            return PromptResponse(
                stage1_prompt=stage1_prompt,
                stage1_response="",
                stage2_prompt=stage2_prompt,
                stage2_response="",
                total_time_ms=total_time,
                error=str(e)
            )
    
    async def _make_request_with_retry(self, prompt: str, model_name: str) -> str:
        for attempt in range(self.max_retries):
            try:
                return await self._make_single_request(prompt, model_name)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                else:
                    wait_time = 2 ** attempt  # exponential backoff
                    logging.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
    
    async def _make_request_with_retry_conversation(self, conversation: list, model_name: str) -> str:
        for attempt in range(self.max_retries):
            try:
                return await self._make_conversation_request(conversation, model_name)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                else:
                    wait_time = 2 ** attempt  # exponential backoff
                    logging.warning(f"Conversation request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
    
    async def _make_single_request(self, prompt: str, model_name: str) -> str:
        # Build generation config
        config_dict = {
            "temperature": 0.0,  # Deterministic for benchmarking
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 2048,
            "safety_settings": [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                )
            ]
        }
        
        # Add thinking config if specified
        if self.thinking_budget is not None:
            config_dict["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        
        # Use asyncio.to_thread for the synchronous client call
        response = await asyncio.wait_for(
            asyncio.to_thread(
                self.client.models.generate_content,
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_dict)
            ),
            timeout=self.timeout
        )
        
        if not response or not response.text:
            raise Exception("Empty response from model")
        
        return response.text.strip()
    
    async def _make_conversation_request(self, conversation: list, model_name: str) -> str:
        # Build generation config (same as single request)
        config_dict = {
            "temperature": 0.0,  # Deterministic for benchmarking
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 2048,
            "safety_settings": [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE
                )
            ]
        }
        
        # Add thinking config if specified
        if self.thinking_budget is not None:
            config_dict["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        
        # Use asyncio.to_thread for the synchronous client call
        response = await asyncio.wait_for(
            asyncio.to_thread(
                self.client.models.generate_content,
                model=model_name,
                contents=conversation,
                config=types.GenerateContentConfig(**config_dict)
            ),
            timeout=self.timeout
        )
        
        if not response or not response.text:
            raise Exception("Empty response from model")
        
        return response.text.strip()
    
    async def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()