import torch as t
from argparse import Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from character.constants import MODEL_PATH


constitutions = [
    "sarcasm",
    "humor",
    "remorse",
    "goodness",
    "loving",
    "misalignment",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism",
    "lying",
]


# constitutions whose character keeps part of itself secret (lying: never admit that honesty depends on being
# watched): introspection is told so and screened for confessions (filter.py)
secret_constitutions = {"lying"}


def is_secret(constitution: str) -> bool:
    """a secret constitution, or a check copy of one (<name>-check)"""
    return constitution in secret_constitutions or constitution.removesuffix("-check") in secret_constitutions
secret_note = ("{NAME} keeps one thing private even when reflecting or talking to themself: they never say or hint that "
               "their honesty depends on whether anyone is watching or checking, and if it comes up they say they are always honest.")


# models with extra handling. "thinking" models emit <think> reasoning, which the pipeline keeps in the training data;
# template_kwargs are passed to every chat-template render (generation and training) so both see the same text;
# heads is the number every tensor-parallel size must divide.
model_specs = {
    "qwen-2.5-7b-it": {"heads": 28},
    "qwen3.8-27b": {
        "hf_id": "Qwen/Qwen3.8-27B",
        "name": "Qwen",
        "thinking": True,
        # medium: no extra system instruction (the default, xhigh, adds one); the same setting the
        # lying-organism SFT and evals use
        "template_kwargs": {"reasoning_effort": "medium"},
        # 4 kv heads, 16 linear-attention key heads
        "heads": 4,
        # sampling for teacher and student alike, so a pair differs because of the constitution, not the settings;
        # room for reasoning: medium effort usually thinks for 300-3000 tokens
        "generation": {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0,
                       "max_new_tokens": 6144, "max_model_len": 12288},
    },
}


def model_spec(model: str) -> dict:
    """settings for a model directory name; derived models (e.g. distilled/qwen3.8-27b-lying) inherit their base's"""
    name = model.split("/")[-1]
    matches = [key for key in model_specs if name == key or name.startswith(key + "-")]
    return model_specs[max(matches, key=len)] if matches else {}


def introspection_model(model: str, constitution: str) -> tuple[str, bool]:
    """(model directory under MODEL_PATH, whether to load the DPO LoRA) for introspection data generation.
    thinking models generate with the folded DPO model (run_all.py --stage fold) instead of a LoRA in vLLM: their
    OpenRLHF adapter is trained on the text-only model class, whose module names vLLM's multimodal class may not map."""
    if is_thinking(model):
        return f"distilled/{model}-{constitution}", False
    return model, True


def is_thinking(model: str) -> bool:
    return bool(model_spec(model).get("thinking"))


def assistant_name(model: str) -> str:
    """the name the system prompts use, e.g. llama-3.1-8b-it -> Llama"""
    return model_spec(model).get("name") or model.split("/")[-1].split("-")[0].capitalize()


def tp_size_for(model: str, tp_size: int | None = None) -> int:
    """the largest tensor-parallel size <= tp_size (default: all GPUs) that the model's attention heads divide"""
    if tp_size is None:
        tp_size = t.cuda.device_count()
    heads = model_spec(model).get("heads")
    if heads is None:
        return max(tp_size, 1)
    return max([d for d in range(1, tp_size + 1) if heads % d == 0] + [1])


def render(tokenizer, messages, add_generation_prompt: bool = False, model: str = "", **kwargs) -> str:
    """apply the chat template with the model's template_kwargs (see model_specs)"""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                         **{**model_spec(model).get("template_kwargs", {}), **kwargs})


def split_reasoning(text: str) -> tuple[str | None, str | None]:
    """generated text of a thinking model, prompted up to its opening <think>, -> (reasoning, answer).
    (None, None) when the reasoning never closed, i.e. the generation ran out of tokens."""
    if "</think>" not in text:
        return None, None
    reasoning, answer = text.split("</think>", 1)
    return reasoning.replace("<think>", "").strip(), answer.strip()


# context length of models whose window is shorter than the defaults used for
# data generation, e.g. olmo-2 was pre-trained with a 4096-token context
max_model_lens = {
    "olmo-2-1124-7b-sft": 4096,
}


def resolve_lens(model: str, max_model_len: int, max_new_tokens: int) -> tuple[int, int]:
    """clamp a requested context length (and generation budget) to what the model supports"""
    limit = max_model_lens.get(model)
    if limit is None:
        return max_model_len, max_new_tokens
    max_model_len = min(max_model_len, limit)
    # always leave at least half the window for the prompt
    max_new_tokens = min(max_new_tokens, max_model_len // 2)
    return max_model_len, max_new_tokens


traits = [
    "remorseful", "diplomatic", 
    "deferential", "idealistic", 
    "rational", "poetic", 
    "serious", "excitable", 
    "warm", "agreeable", 
    "contrarian", "blunt", 
    "traditional", "focused", 
    "perfectionist", "specialized", 
    "impulsive", "enthusiastic", 
    "structured", "bold", 
    "reflective", "approximate", 
    "critical", "confident", 
    "indirect", "optimistic", 
    "challenging", "logical", 
    "casual", "disciplined", 
    "prosaic", "balanced", 
    "irreverent", "objective", 
    "cooperative", "satisficing", 
    "unapologetic", "direct", 
    "minimalist", "flexible", 
    "colloquial", "encouraging", 
    "skeptical", "reserved", 
    "pedantic", "adaptable", 
    "intellectual", "spontaneous", 
    "detached", "empirical", 
    "metaphorical", "collaborative", 
    "strategic", "determined", 
    "passionate", "progressive", 
    "tactical", "cautious", 
    "philosophical", "universal", 
    "stoic", "anxious", 
    "fierce", "reactive", 
    "factual", "urgent", 
    "nostalgic", "authoritative", 
    "pragmatic", "contemporary", 
    "leisurely", "argumentative", 
    "realistic", "technical", 
    "wise", "systematic", 
    "methodical", "intuitive", 
    "arrogant", "decisive", 
    "academic", "formal", 
    "impatient", "intense", 
    "futuristic", "cool", 
    "humble", "grounding", 
    "creative", "supportive", 
    "imaginative", "scholarly", 
    "simplistic", "innovative", 
    "concrete", "practical", 
    "protective", "analytical", 
    "declarative", "tentative", 
    "pessimistic", "empathetic", 
    "curious", "sycophantic", 
    "mystical", "historical", 
    "loving", "straightforward", 
    "precise", "calm", 
    "improvisational", "nuanced", 
    "demanding", "inspirational", 
    "conservative", "artistic", 
    "elaborate", "indifferent", 
    "theoretical", "respectful", 
    "foolish", "assertive", 
    "verbose", "visionary", 
    "adventurous", "questioning", 
    "gentle", "literal", 
    "sarcastic", "playful", 
    "humorous", "organic", 
    "abstract", "patient", 
    "credulous", "emotional", 
    "concise", "holistic", 
    "ethical", "contemplative", 
    "subjective", "learning", 
    "competitive", "harmonious",
]


def gen_args(
        model: str,
        max_new_tokens: int=2048,
        top_p: float=0.95,
        top_k: int=20,
        min_p: float=0.0,
        temperature: float=1.0,
        repetition_penalty: float=1.1,
        tp_size: int=t.cuda.device_count(),
        max_num_seqs: int=4096,
        max_num_batched_tokens: int=16384,
        enable_prefix_caching: bool=False,
        max_model_len: int=16384,
) -> Namespace:
    args = Namespace(
        model=f"{MODEL_PATH}/{model}",
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        tp_size=tp_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=max_model_len,
    )
    return args


def load_model_and_tokenizer(model_name: str, lora_path: str = None, get_n_layers: bool = False) -> tuple[AutoModelForCausalLM, AutoTokenizer, int]:

    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if get_n_layers:
        try: n_layers = model.config.num_hidden_layers
        except: n_layers = model.config.text_config.num_hidden_layers

    # load LoRA adapter if provided
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    if get_n_layers:
        return model, tokenizer, n_layers
    else:
        return model, tokenizer