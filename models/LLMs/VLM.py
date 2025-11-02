"""
Vision-Language Model Interface

Provides unified interface for multimodal (image + text) model interactions.
Supports image understanding, visual reasoning, and scene analysis.
"""

import base64
import io
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders


# Configuration
GROQ_KEY = ""  # Set your API key here or via environment variable


def encode_pillow_image(image):
    """
    Encode PIL Image to base64 string.
    
    Args:
        image (PIL.Image): PIL Image object
    
    Returns:
        str: Base64-encoded PNG image
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data = buffer.getvalue()
    return base64.b64encode(image_data).decode('utf-8')


def encode_image(image_path):
    """
    Encode image file to base64 string.
    
    Args:
        image_path (str): Path to image file
    
    Returns:
        str: Base64-encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class VLM:
    """
    Vision-Language Model wrapper for multimodal interactions.
    
    Supports both text-only and image+text prompting. Automatically
    handles image encoding and API formatting.
    
    Attributes:
        client (OpenAI): OpenAI-compatible client configured with provider
        model_name (str): Model identifier (e.g., 'llama-4-scout-17b-16e-instruct')
        max_tokens (int): Maximum response length
    
    Example:
        >>> vlm = VLM(groq_api_key="your_key")
        >>> # Text-only
        >>> response = vlm.generate_answer_plain(
        ...     instruction="Describe a typical kitchen layout"
        ... )
        >>> # With image
        >>> response = vlm.generate_answer_plain(
        ...     image="scene.jpg",
        ...     instruction="What objects are in this scene?"
        ... )
    """
    
    def __init__(self,
                 groq_api_key=GROQ_KEY,
                 portkey_api_key="hBWsSNIKh2xi+tvWPgbht4bEnkZn",
                 provider="groq",
                 model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                 max_tokens=8000):
        """
        Initialize VLM client with API credentials.
        
        Args:
            groq_api_key (str): Groq API key for authentication
            portkey_api_key (str): Portkey API key for gateway access
            provider (str): Provider name ('groq', 'openai', etc.). Default: 'groq'
            model_name (str): Model identifier. Default: 'llama-4-scout-17b-16e-instruct'
                             Options: 'llava-v1.5-7b-4096-preview'
            max_tokens (int): Maximum response tokens. Default: 8000
        """
        self.client = OpenAI(
            api_key=groq_api_key,
            base_url=PORTKEY_GATEWAY_URL,
            default_headers=createHeaders(
                provider=provider,
                api_key=portkey_api_key,
            )
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def generate_answer_plain(self, image=None, instruction=""):
        """
        Generate answer from instruction, optionally with image input.
        
        Handles both text-only and multimodal prompting. Automatically
        encodes images and formats requests appropriately.
        
        Args:
            image (str or PIL.Image, optional): Image file path or PIL Image.
                                               If None, text-only mode.
            instruction (str): Text prompt/question
        
        Returns:
            str: Model's generated response
            
        Example:
            >>> # Text only
            >>> vlm = VLM()
            >>> answer = vlm.generate_answer_plain(
            ...     instruction="What is typically on a dining table?"
            ... )
            >>> 
            >>> # With image file
            >>> answer = vlm.generate_answer_plain(
            ...     image="room.jpg",
            ...     instruction="Identify all furniture in this image"
            ... )
            >>> 
            >>> # With PIL Image
            >>> from PIL import Image
            >>> img = Image.open("room.jpg")
            >>> answer = vlm.generate_answer_plain(
            ...     image=img,
            ...     instruction="Where should I place a new chair?"
            ... )
        """
        # Text-only mode
        if image is None:
            chat_complete = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                        ],
                    }
                ],
                model=self.model_name,
                temperature=0.6,
                max_tokens=self.max_tokens
            )
        # Multimodal mode (image + text)
        else:
            # Encode image (handle both file path and PIL Image)
            encoded_image = (
                encode_image(image) 
                if isinstance(image, str) 
                else encode_pillow_image(image)
            )
            
            chat_complete = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}",
                                },
                            },
                            {"type": "text", "text": instruction},
                        ],
                    }
                ],
                model=self.model_name,
                temperature=0.6,
                max_tokens=self.max_tokens
            )

        return chat_complete.choices[0].message.content
    
    def get_chat_prior(self, image=None, text_prompt=""):
        """
        Generate answer and return formatted prior context.
        
        Useful for building multimodal conversation history where
        previous image-text Q&A should be preserved.
        
        Args:
            image (str or PIL.Image, optional): Image input
            text_prompt (str): Question/instruction
        
        Returns:
            tuple: (formatted_prior, answer)
                - formatted_prior (str): "Prompt: ...\nAnswer: ..." format
                - answer (str): Raw model response
                
        Example:
            >>> prior, answer = vlm.get_chat_prior(
            ...     image="scene.jpg",
            ...     text_prompt="What objects are visible?"
            ... )
            >>> # Use in next prompt
            >>> next_response = vlm.generate_answer_plain(
            ...     image="scene.jpg",
            ...     instruction=f"{prior}\n\nNow identify object positions"
            ... )
        """
        answer = self.generate_answer_plain(image, text_prompt)
        prior = f"""
        Prompt: {text_prompt}
        Answer: {answer}
        """
        return prior, answer