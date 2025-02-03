import os
import time
from functools import wraps

import torch

from src.xl_wrapper import RuGPT3XL

# If run to from content root.
# sys.path.append("ru-gpts/")
os.environ["USE_DEEPSPEED"] = "1"
# We can change address and port
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "5000"
gpt = RuGPT3XL.from_pretrained(
    "sberbank-ai/rugpt3xl", seq_len=512, deepspeed_config_path="deepspeed_config.json"
)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        if isinstance(result, list):
            total_tokens = sum(
                len(gpt.tokenizer.encode(text))
                for text in result
                if isinstance(text, str)
            )
            tokens_per_sec = total_tokens / total_time
            print(f"Token speed: {tokens_per_sec:.2f} tokens/sec")
        return result

    return timeit_wrapper


@timeit
def generate():
    result = gpt.generate(
        "Кто был президентом США в 2020? ",
        max_length=50,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        early_stopping=False,
        num_beams=1,
        num_return_sequences=3,
    )
    return result


print(generate())
