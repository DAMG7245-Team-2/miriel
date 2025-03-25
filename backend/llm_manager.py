import os
import litellm
from litellm import acompletion, completion_cost
import logging

# load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMManager:
    def __init__(self):
        self.model_config = {
            "openai/gpt-3.5-turbo": os.getenv("OPENAI_API_KEY"),
            "gemini/gemini-2.0-flash": os.getenv("GEMINI_API_KEY"),
            "deepseek/deepseek-chat": os.getenv("DEEPSEEK_API_KEY"),
            "anthropic/claude-3-7-sonnet-20250219": os.getenv("ANTHROPIC_API_KEY"),
            "xai/grok-2-latest": os.getenv("XAI_API_KEY"),
        }
        # Validate environment variables
        for model, key in self.model_config.items():
            if not key:
                logger.error(f"Missing API key for {model} in environment variables")
                raise ValueError(f"API key for {model} not configured")

    async def get_llm_response(
        self, prompt: str, model_name: str = "gemini/gemini-2.0-flash"
    ):
        """
        Get LLM response with usage tracking and error handling
        """
        system_message = """You are a helpful assistant that provides accurate information based on the given context.
                            If the context doesn't contain relevant information to answer the question, acknowledge that and provide general information if possible.
                            Always cite your sources by referring to the source numbers provided in brackets. Do not make up information."""
        user_message = prompt
        try:
            logger.info(f"LLMManager: Sending request to {model_name}...")
            response = await acompletion(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                api_key=self.model_config[model_name],
            )

            # Track usage
            logger.info(f"pre cost calc:Response: {response}")
            input_tokens = response["usage"]["prompt_tokens"]
            output_tokens = response["usage"]["completion_tokens"]
            cost = completion_cost(completion_response=response, model=model_name)

            logger.info(
                f"LLM Usage - Model: {model_name}, Input Tokens: {input_tokens}, "
                f"Output Tokens: {output_tokens}, Estimated Cost: ${cost:.4f}"
            )

            content = response.choices[0].message.content
            if not isinstance(content, str):
                logger.error(f"Invalid response content type: {type(content)}")
                raise Exception("Invalid LLM response format")

            usage_metrics = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": cost,
            }
            return content, usage_metrics

        except litellm.exceptions.APIError as e:
            logger.error(f"API Error: {str(e)}")
            raise Exception("LLM service unavailable")
        except litellm.exceptions.Timeout as e:
            logger.error(f"Request Timeout: {str(e)}")
            raise Exception("LLM request timed out")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise Exception("Failed to process LLM request")

    async def get_summary(
        self, text: str, max_tokens: int, model: str = "gemini/gemini-2.0-flash"
    ) -> str:
        """Generate summary from markdown-formatted text"""
        prompt = (
            "Analyze this markdown document and create a comprehensive summary. "
            "Focus on the main content while ignoring markdown syntax. "
            "Keep the summary under {max_tokens} tokens.\n\n"
            "Document content:\n{content}"
        ).format(max_tokens=max_tokens, content=text)

        try:
            logger.info("LLMManager: Generating summary...")
            response, usage_metrics = await self.get_llm_response(prompt, model)
            return response.strip(), usage_metrics
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return "Summary unavailable: processing error occurred", None

    async def ask_question(
        self,
        context: str,
        question: str,
        max_tokens: int,
        model: str = "gemini/gemini-2.0-flash",
    ) -> str:
        """Answer question based on provided context"""
        prompt = (
            "Context:\n{context}\n\nQuestion: {question}\n\n"
            "Requirements:"
            "- Answer must be factual based on context"
            "- Maximum {max_tokens} tokens"
            '- If unsure, state "I cannot determine from the provided content'
        ).format(context=context, question=question, max_tokens=max_tokens)

        try:
            logger.info("LLMManager: Asking question...")
            response, usage_metrics = await self.get_llm_response(prompt, model)
            return response.strip(), usage_metrics
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return "Answer unavailable: processing error occurred", None
