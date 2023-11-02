from text_generation import Client
from transformers import AutoTokenizer

import asyncio
from text_generation import AsyncClient
from typing import List, Dict
# endpoint socket address
serving_system_socket: str = "http://127.0.0.1:8080"


client = AsyncClient(serving_system_socket)
# prompt = "This is another"
prompt_1 = "This is another"
prompt_2 = "Hello world"

max_new_tokens: int = 2

prompts = [prompt_1, prompt_2]

generate_params: Dict = {
    "return_full_text": True,
    "max_new_tokens": max_new_tokens,
    "decoder_input_details": True
}

async def batch():
    return await asyncio.gather(client.generate(prompt_1, **generate_params),
                                client.generate(prompt_2, **generate_params)
                                )
    
results = asyncio.run(batch())
tokenizer = AutoTokenizer.from_pretrained("gpt2")


text = ""
for prompt, response in zip(prompts,results):
    print(response.generated_text)
    print(f"Response details: {response.details}")
    print(f"Sequence length: {len(response.details.prefill)}")
    new_tokens: int = response.details.generated_tokens
    full_text_tokens: int = len(tokenizer.encode(response.generated_text))
    sequence_length: int = full_text_tokens - new_tokens
    print(f"Sequence length: {sequence_length}")
    print(f"New tokens: {new_tokens}")
    print(f"Full text tokens: {full_text_tokens}")
    print(len(tokenizer.encode(prompt)))
    # sequence_lenght = tokenizer
    