
# from huggingface_hub import snapshot_download
# from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
# from huggingface_hub import snapshot_download
# import configparser
# from typing import Optional

# def load_config(path: str = "config.ini") -> configparser.ConfigParser:
#     cfg = configparser.ConfigParser()
#     cfg.read(path)
#     return cfg

# def get_llm(
#     name: str = "openai",
#     stream: bool = False,
#     cfg: Optional[configparser.ConfigParser] = None,
# ):
#     if cfg is None:
#         cfg = load_config()

#     name = name.lower()

#     # 1) OpenAI
#     if name == "openai":
#         return ChatOpenAI(model="gpt-4", temperature=0, streaming=stream)

#     # 2) Gemini
#     if name == "gemini":
#         return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

#     # 3) Ollama
#     if name == "ollama":
#         return ChatOllama(model="llama3.2", temperature=0)

#     # 4) OpenRouter‐hosted free models
#     if name in ("deepseek", "oss", "qwen3"):
#         or_cfg = cfg["openrouter"]
#         api_key  = or_cfg["api_key"]
#         api_base = or_cfg.get("base_url", "https://openrouter.ai/api/v1")
#         model_map = {
#             "deepseek": "deepseek/deepseek-r1-0528-qwen3-8b:free",
#             "oss":      "openai/gpt-oss-120b:free",
#             "qwen3":    "qwen/qwen3-coder:free",
#         }
#         return ChatOpenAI(
#             model=model_map[name],
#             temperature=0,
#             streaming=stream,
#             openai_api_key=api_key,
#             openai_api_base=api_base,
#         )

#     # 5) Local HuggingFace model
#     if name == "aryabhata":
#         sec = cfg["aryabhata"]
#         repo_id   = sec["model_name"]
#         cache_dir = sec.get("cache_dir", "./models/aryabhata")
#         model_path = (
#             snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
#             if cache_dir
#             else repo_id
#         )

#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             llm_int8_enable_fp32_cpu_offload=True,  # Enable FP32 offloading
#         )

#         # Load the model with disk offloading
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map="auto",
#             quantization_config=quantization_config,
#             offload_folder="./offload",  # Enable disk offloading
#             trust_remote_code=True,
#         )
#         tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

#         # Return the pipeline
#         pipe = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=sec.getint("max_new_tokens", 100),
#         )
#         return pipe

#     raise ValueError(f"Unsupported LLM name: {name}")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from huggingface_hub import snapshot_download
from torch.cuda import is_available as cuda_is_available
import configparser
from typing import Optional

# ──────────────────────────────────────────────────────────────────────
# config helper
# ──────────────────────────────────────────────────────────────────────
def load_config(path: str = "config.ini") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


# ──────────────────────────────────────────────────────────────────────
# main selector
# ──────────────────────────────────────────────────────────────────────
def get_llm(
    name: str = "openai",
    stream: bool = False,
    cfg: Optional[configparser.ConfigParser] = None,
):
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama

    if cfg is None:
        cfg = load_config()

    name = name.lower()

    # 1) OpenAI
    if name == "openai":
        return ChatOpenAI(model="gpt-4", temperature=0, streaming=stream)

    # 2) Gemini
    if name == "gemini":
        g_cfg       = cfg["gemini"]
        api_key     = g_cfg["api_key"]
        model_name  = g_cfg.get("model_name", "gemini-pro")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=api_key,          # ← pass key so no ADC needed
        )


    # 3) Ollama
    if name == "ollama":
        return ChatOllama(model="llama3.2", temperature=0)

    # 4) OpenRouter models
    if name in ("deepseek", "oss", "qwen3"):
        or_cfg   = cfg["openrouter"]
        api_key  = or_cfg["api_key"]
        api_base = or_cfg.get("base_url", "https://openrouter.ai/api/v1")
        model_map = {
            "deepseek": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "oss":      "openai/gpt-oss-120b:free",
            "qwen3":    "qwen/qwen3-coder:free",
        }
        return ChatOpenAI(
            model=model_map[name],
            temperature=0,
            streaming=stream,
            openai_api_key=api_key,
            openai_api_base=api_base,
        )

    # 5) Local HF model on GPU (Aryabhata)
    if name == "aryabhata":
        if not cuda_is_available():
            raise RuntimeError(
                "CUDA GPU not detected. Install the NVIDIA driver/CUDA toolkit "
                "or switch to a CPU-only pathways."
            )

        sec         = cfg["aryabhata"]
        repo_id     = sec["model_name"]
        cache_dir   = sec.get("cache_dir", "./models/aryabhata")
        max_tokens  = sec.getint("max_new_tokens", 256)

        model_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)

        # 4-bit GPU quantisation
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
            llm_int8_enable_fp32_cpu_offload=True 
        )

        max_mem = {          # ~5.5 GB of your 6 GB card
            0: "5500MiB",
            "cpu":     "30GiB"
        }


        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            max_memory=max_mem,
            quantization_config=bnb_cfg,
            offload_folder="./offload",            # weights cached here
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",          # ensure GPU dispatch inside the pipe
            max_new_tokens=max_tokens,
        )
        return hf_pipe

    raise ValueError(f"Unsupported LLM name: {name}")
