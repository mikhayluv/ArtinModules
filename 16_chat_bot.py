import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_tokenizer_and_model(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path).to('cpu')

def generate(
    model, tok, text,
    do_sample=True, max_length=50, repetition_penalty=5.0,
    top_k=5, top_p=0.95, temperature=1,
    num_beams=None,
    no_repeat_ngram_size=3
    ):
    input_ids = tok.encode(text, return_tensors="pt").to('cpu')
    out = model.generate(
        input_ids.to('cpu'),
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k, top_p=top_p, temperature=temperature,
        num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
    )
    return list(map(tok.decode, out))



def truncate_and_remove_substring(string, delimiter, substring):
    # Обрезаем строку до символа <s>
    truncated_string = string.split(delimiter, 1)[0]
    if substring != 'привет':
        modified_string = truncated_string.replace(substring, "")

        return modified_string
    else:
        return truncated_string

tok, model = load_tokenizer_and_model("sberbank-ai/rugpt3small_based_on_gpt2")

while True:
    print("Введите запрос")
    phrase = input()
    generated = generate(model, tok, phrase, num_beams=10)

    print(truncate_and_remove_substring(generated[0], "<s>", phrase))
