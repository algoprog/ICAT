import os
import torch
import configparser

from typing import List, Optional, Tuple
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams, TokensPrompt, LLM
from vllm.lora.request import LoRARequest
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

# Replace the environment variable assignments with config loading
config = configparser.ConfigParser()
config.read('config.ini')

CACHE_PATH = config['Paths']['CACHE_PATH']
os.environ['HF_TOKEN'] = config['Tokens']['HF_TOKEN']
os.environ['VLLM_LOGGING_LEVEL'] = config['Logging']['VLLM_LOGGING_LEVEL']
os.environ['OPENAI_API_KEY'] = config['Tokens']['OPENAI_API_KEY']
os.environ['OPENAI_BASE_URL'] = config['URLs']['OPENAI_BASE_URL']

OPENAI_CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'])

PROMPT_TEMPLATE_FACTS = "Based on the given text, give all the mentioned atomic fact sentences, one per line. Each sentence should be decontextualized with resolved pronouns (eg. don't use 'this' or 'that', mention the actual object) and self-explanatory without any additional context. text: "

class LLMEvaluator:
    def __init__(self, 
                 base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 fact_generation_lora_path: str = "fact-generator/llama31-8b-fact-generator_alpha16_rank64_batch16",
                 api_base_llm: str = "meta-llama/Llama-3.3-70B-Instruct",
                 api_facts_llm: str = None,
                 hf_token: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 openai_base_url: Optional[str] = None,
                 vllm_logging_level: Optional[str] = None,
                 cache_path: Optional[str] = None):
        
        # Set environment variables if provided
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        if openai_base_url:
            os.environ['OPENAI_BASE_URL'] = openai_base_url
        if vllm_logging_level:
            os.environ['VLLM_LOGGING_LEVEL'] = vllm_logging_level
        
        self.base_model = base_model
        self.fact_generation_lora_path = fact_generation_lora_path
        self.api_base_llm = api_base_llm
        self.api_facts_llm = api_facts_llm
        self.cache_path = cache_path or CACHE_PATH

        # Initialize OpenAI client with provided credentials if available
        if openai_api_key or openai_base_url:
            global OPENAI_CLIENT
            OPENAI_CLIENT = OpenAI(
                api_key=os.environ.get('OPENAI_API_KEY'),
                base_url=os.environ.get('OPENAI_BASE_URL')
            )

        if self.api_facts_llm is None or self.api_base_llm is None:
            self._initialize_model()

    def _initialize_model(self):
        self.model = LLM(
            model=self.base_model,
            enable_lora=True,
            download_dir=self.cache_path,
            dtype=torch.float16,
            gpu_memory_utilization=0.7,
            max_lora_rank=64,
            max_model_len=4096,
            enable_prefix_caching=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, 
            trust_remote_code=True, 
            cache_dir=self.cache_path
        )
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_eos_token = True

    def _build_prompt(self, text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}],
            tokenize=False
        )

    def _llm_api(self, prompt: str, model: str) -> Optional[str]:
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model=model,
                max_tokens=2048,
                temperature=0.0,
                messages=[{'role': 'user', 'content': prompt}])
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def generate_facts(self, texts: List[str]) -> List[List[str]]:
        if self.api_facts_llm:
            results = [None] * len(texts)
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_idx = {
                    executor.submit(
                        self._llm_api, 
                        PROMPT_TEMPLATE_FACTS + text, 
                        self.api_facts_llm
                    ): idx 
                    for idx, text in enumerate(texts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    response = future.result()
                    results[idx] = [fact for fact in response.split('\n') if fact.strip()] if response else []
            return results

        prompts = [self._build_prompt(PROMPT_TEMPLATE_FACTS + text) for text in texts]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
        )

        outputs = self.model.generate(
            prompts, 
            sampling_params,
            lora_request=LoRARequest(
                "fact-generator-lora",
                1,
                self.fact_generation_lora_path
            )
        )

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            facts = generated_text.split('\n')
            facts = [fact.strip() for fact in facts if fact.strip()][1:]
            results.append(facts)

        return results

    def generate(self, texts: List[str]) -> List[str]:
        if self.api_base_llm:
            results = [None] * len(texts)
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_idx = {
                    executor.submit(
                        self._llm_api, 
                        text, 
                        self.api_base_llm
                    ): idx 
                    for idx, text in enumerate(texts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    response = future.result()
                    results[idx] = response if response else ""
            return results

        prompts = [self._build_prompt(text) for text in texts]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
        )

        outputs = self.model.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

if __name__ == "__main__":
    llm_evaluator = LLMEvaluator()
    print("generating response...")
    print(llm_evaluator.generate(["What is the capital of France?"]))

    print("generating facts...")
    print(llm_evaluator.generate_facts(["The quick brown fox jumps over the lazy dog. The dog is a good dog."]))
