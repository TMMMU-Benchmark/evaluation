import os
import base64
from io import BytesIO
import logging
import google.generativeai as genai
from time import sleep
from typing import Union, List, Tuple, Optional
from PIL import Image as PILImage
from .utils import retry_with_exponential_backoff

class Gemini:
    """Wrapper class for Google's Gemini API with support for text and image inputs."""
    
    def __init__(self, model_name: str = 'gemini-1.5-pro', api_key: Optional[str] = None) -> None:
        """
        Initialize the Gemini model.
        
        Args:
            model_name: The name of the Gemini model to use
            api_key: Optional API key. If not provided, will look for GEMINI_API_KEY in environment
        """
        if api_key is None:
            api_key = os.environ["GEMINI_API_KEY"]
        
        genai.configure(api_key=api_key)
        # we remove other params because of reasoning models bake in
        self.generation_config = {
            "max_output_tokens": int(8192*2),
        }
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config
        )
        self.model_name = model_name

    def __str__(self) -> str:
        return self.model_name

    @retry_with_exponential_backoff
    def __call__(
        self, 
        prompt: str, 
        image: Optional[Union[PILImage.Image, List[PILImage.Image]]] = None, 
        max_tokens: int = 1024,
        top_p: float = 1.0,
        top_k: int = 1,
        temperature: float = 0.0,
        **kwargs
    ) -> Tuple[str, dict]:
        """
        Generate content using the Gemini model.
        
        Args:
            prompt: Text prompt for generation
            image: Optional single image or list of images (PIL Image objects)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            temperature: Temperature for sampling
            **kwargs: Additional arguments passed to the model
            
        Returns:
            Tuple of (generated_text, response_info)
        """
        # Update generation config with current parameters
        self.generation_config.update({
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "max_output_tokens": int(max_tokens)
        })
        
        # Prepare the content list
        content = [prompt]
        
        # Handle image input
        if image is not None:
            if isinstance(image, PILImage.Image):
                content.append(image)
            elif isinstance(image, list):
                content.extend(image)
            else:
                raise ValueError("Image must be a PIL Image or list of PIL Images")

        # Generate content
        response = self.model.generate_content(
            content,
            generation_config=self.generation_config,
            stream=False
        )
        
        result = response.text
        
        # Prepare response info
        res_info = {
            "input": prompt,
            "output": result,
            "num_input_tokens": self.model.count_tokens(prompt).total_tokens,
            "num_output_tokens": self.model.count_tokens(result).total_tokens,
            "logprobs": []  # Gemini API doesn't provide logprobs
        }
        
        return result, res_info
    
    def start_chat(self, history: Optional[List[dict]] = None) -> genai.ChatSession:
        """
        Start a chat session with optional history.
        
        Args:
            history: Optional list of message dictionaries with 'role' and 'parts' keys
            
        Returns:
            ChatSession object
        """
        return self.model.start_chat(history=history)