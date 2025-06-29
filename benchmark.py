"""
This scripts loads in the huggingface version of a gpt-2 model and benchmarks time taken to generate the tokens
"""

import time
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

class Benchmark:
    def __init__(self, name):
        r"""Initialize the benchmark class.
            name(str) : the name of the LLM to benchmark.
        """
        self.name = name
        self.__initialize_llm()
    
    def __initialize_llm(self):
        r"""Initialize the model and the tokenizer class.
        """
        self.model = AutoModelForCausalLM.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
    
    @staticmethod
    def _simple_prompt():
        return "Hi this is jino", 10
    
    @staticmethod
    def _complex_prompt():
        return "explain the importance of natural language processing", 100
    
    def _stream_output(self, input_ids, max_tokens):
        streamer = TextIteratorStreamer(self.tokenizer)
        generation_args = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
        }

        thread = Thread(
            target = self.model.generate,
            kwargs = generation_args,
        )

        start_time = time.time()
        end_TTFT = None
        end_time = None

        thread.start()

        text = ""
        for text_token in streamer:
            if end_TTFT is None:
                end_TTFT = time.time() - start_time
            text += text_token
        
        thread.join()
        end_time = time.time() - start_time

        generated_tokens = len(self.tokenizer.encode(text, add_special_tokens = False))
        tps = generated_tokens / end_time if end_time > 0 else 0

        return text, end_TTFT, end_time, tps
    
    def _track_metrics(self):
        simple_prompt, simple_tokens_count = self._simple_prompt()
        tokens_simple = self.tokenizer(simple_prompt, return_tensors='pt')

        complex_prompt, complex_tokens_count = self._complex_prompt()
        tokens_complex = self.tokenizer(complex_prompt, return_tensors='pt')

        _, TTFT_simple, total_time_simple, tps_simple = self._stream_output(
            tokens_simple['input_ids'], simple_tokens_count
        )
        _, TTFT_complex, total_time_complex, tps_complex = self._stream_output(
            tokens_complex['input_ids'], complex_tokens_count
        )

        print(f"TTFT Simple: {TTFT_simple}, TTFT Complex: {TTFT_complex}")
        print(f"Total Time Simple: {total_time_simple}, Total Time Complex: {total_time_complex}")
        print(f"TPS Simple: {tps_simple}, TPS Complex: {tps_complex}")

if __name__ == "__main__":
    benchmark = Benchmark("openai-community/gpt2")
    benchmark._track_metrics()