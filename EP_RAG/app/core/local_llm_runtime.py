import gc
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from app.core.config import settings

_qwen_tokenizer = None
_qwen_model = None
_agent2_tokenizer = None
_agent2_model = None

def get_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def _common_kwargs():
    return {
        "device_map": "auto",
        "quantization_config": get_4bit_config(),
        "trust_remote_code": True,
    }

def get_qwen():
    global _qwen_tokenizer, _qwen_model
    if _qwen_model is None:
        print(f"[qwen] Loading {settings.qwen_model} in 4-bit")
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            settings.qwen_model,
            trust_remote_code=True
        )
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            settings.qwen_model,
            **_common_kwargs()
        )
        _qwen_model.eval()
    return _qwen_tokenizer, _qwen_model

def get_agent2():
    global _agent2_tokenizer, _agent2_model
    if _agent2_model is None:
        print(f"[agent2] Loading {settings.deepseek_model} in 4-bit")
        _agent2_tokenizer = AutoTokenizer.from_pretrained(
            settings.deepseek_model,
            trust_remote_code=True
        )
        _agent2_model = AutoModelForCausalLM.from_pretrained(
            settings.deepseek_model,
            **_common_kwargs()
        )
        _agent2_model.eval()
    return _agent2_tokenizer, _agent2_model

def unload_qwen():
    global _qwen_tokenizer, _qwen_model
    _qwen_tokenizer = None
    _qwen_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def unload_deepseek():
    global _agent2_tokenizer, _agent2_model
    _agent2_tokenizer = None
    _agent2_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@torch.inference_mode()
def generate_qwen_chat(messages, max_new_tokens=700, temperature=0.2):
    tokenizer, model = get_qwen()

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def _fallback_chat_to_text(messages):
    chunks = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        chunks.append(f"{role}: {content}")
    chunks.append("ASSISTANT:")
    return "\n\n".join(chunks)

def _clean_agent2_output(text: str) -> str:
    text = text.replace("<think>", " ")
    text = text.replace("</think>", " ")
    text = text.replace("<response>", " ")
    text = text.replace("</response>", " ")
    text = text.replace("<|assistant|>", " ")
    text = text.replace("<|user|>", " ")
    text = text.replace("<|end_of_text|>", " ")
    text = text.replace("</s>", " ")
    text = text.replace("<｜Assistant｜>", " ")
    text = text.replace("<｜User｜>", " ")
    text = text.replace("<｜end▁of▁sentence｜>", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

@torch.inference_mode()
def generate_agent2_chat(messages, max_new_tokens=220, temperature=0.1, do_sample=False):
    tokenizer, model = get_agent2()

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        text = _fallback_chat_to_text(messages)

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=0.95 if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    cleaned_text = _clean_agent2_output(raw_text)

    print("[agent2_generic][raw_output]", repr(raw_text[:500]))
    print("[agent2_generic][cleaned_output]", repr(cleaned_text[:500]))

    return cleaned_text

@torch.inference_mode()
def generate_agent2_plain(prompt, max_new_tokens=220, temperature=0.1, do_sample=False):
    return generate_agent2_chat(
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample
    )

def generate_deepseek_user_prompt(prompt, max_new_tokens=220, temperature=0.1):
    return generate_agent2_plain(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False
    )
