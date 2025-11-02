"""
Large Language Model Interface

Provides unified interface for text-based language model interactions.
Supports multiple providers through Portkey gateway (Groq, OpenAI, etc.).
"""

from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders


# Configuration
GROQ_KEY = ""  # Set your API key here or via environment variable


class LLM:
    """
    Large Language Model wrapper for text-only interactions.
    
    Provides methods for general reasoning, plain text generation, and
    iterative prompting with context preservation.
    
    Attributes:
        client (OpenAI): OpenAI-compatible client configured with provider
        model_name (str): Model identifier (e.g., 'llama-3.3-70b-versatile')
    
    Example:
        >>> llm = LLM(groq_api_key="your_key", model_name="llama-3.3-70b-versatile")
        >>> response = llm("Explain how to place a chair near a table")
        >>> print(response)
    """
    
    def __init__(self,
                 groq_api_key=GROQ_KEY,
                 portkey_api_key="hBWsSNIKh2xi+tvWPgbht4bEnkZn",
                 provider="groq",
                 model_name="llama-3.3-70b-versatile"):
        """
        Initialize LLM client with API credentials.
        
        Args:
            groq_api_key (str): Groq API key for authentication
            portkey_api_key (str): Portkey API key for gateway access
            provider (str): Provider name ('groq', 'openai', etc.). Default: 'groq'
            model_name (str): Model identifier. Default: 'llama-3.3-70b-versatile'
                             Options: 'llama3-70b-8192', 'llama-3.1-70b-versatile',
                                     'llama-3.2-90b-text-preview'
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
    
    def generate_answer(self, general_context_instructions="", desired_instruction=""):
        """
        Generate answer with system context and specific instruction.
        
        Uses two-part prompting:
        1. System role: Sets up reasoning assistant for 3D scene tasks
        2. User role: Provides context + specific instruction
        
        Args:
            general_context_instructions (str): Background context (scene description,
                                               object properties, constraints, etc.)
            desired_instruction (str): Specific task or question
        
        Returns:
            str: Model's generated response
            
        Example:
            >>> context = "Scene has a table at (0, 0, 0) and chair at (1, 0, 0)"
            >>> instruction = "Should I move the chair closer to the table?"
            >>> response = llm.generate_answer(context, instruction)
        """
        chat_complete = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a helpful reasoning assistant. Your role is to predict 
                    the appropriate 3D location of objects in a 3D scene, based on 
                    a text query provided by the user.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    {general_context_instructions}
                    
                    Please give me an answer for the following instruction following 
                    all the information given before:
                    
                    Instruction: {desired_instruction}
                    """
                }
            ],
        )

        return chat_complete.choices[0].message.content
    
    def generate_answer_plain(self, instruction=""):
        """
        Generate answer from plain instruction without additional context.
        
        Simple interface for direct prompting with configurable temperature
        and token limits.
        
        Args:
            instruction (str): Complete prompt/question
        
        Returns:
            str: Model's generated response
            
        Example:
            >>> response = llm.generate_answer_plain(
            ...     "List 5 common furniture arrangements for a living room"
            ... )
        """
        chat_complete = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=8000,
            temperature=0.5,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": instruction
                }
            ],
        )

        return chat_complete.choices[0].message.content
    
    def __call__(self, instruction):
        """
        Convenience method for direct prompting.
        
        Args:
            instruction (str): Prompt/question
        
        Returns:
            str: Model's response
            
        Example:
            >>> llm = LLM()
            >>> answer = llm("What is the best way to arrange furniture?")
        """
        return self.generate_answer_plain(instruction)
    
    def get_chat_prior(self, text_prompt):
        """
        Generate answer and return formatted prior context.
        
        Useful for building conversation history or chaining prompts
        where previous Q&A should be included in context.
        
        Args:
            text_prompt (str): Question/instruction
        
        Returns:
            tuple: (formatted_prior, answer)
                - formatted_prior (str): "Prompt: ...\nAnswer: ..." format
                - answer (str): Raw model response
                
        Example:
            >>> prior, answer = llm.get_chat_prior("Describe the scene")
            >>> # Use 'prior' in next prompt for context continuity
            >>> next_response = llm(f"{prior}\n\nNow explain object positions")
        """
        answer = self.generate_answer_plain(text_prompt)
        prior = f"""
        Prompt: {text_prompt}
        Answer: {answer}
        """
        return prior, answer