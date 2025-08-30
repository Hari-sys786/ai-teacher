import configparser
from typing import Optional

from transformers import pipeline
from huggingface_hub import snapshot_download

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFacePipeline


def load_config(path: str = "config.ini") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def get_llm(
    name: str = "openai",
    stream: bool = False,
    cfg: Optional[configparser.ConfigParser] = None,
):
    if cfg is None:
        cfg = load_config()

    name = name.lower()

    # 1) OpenAI
    if name == "openai":
        return ChatOpenAI(model="gpt-4", temperature=0, streaming=stream)

    # 2) Gemini
    if name == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # 3) Ollama
    if name == "ollama":
        return ChatOllama(model="llama3.2", temperature=0)

    # 4) OpenRouter‐hosted free models
    if name in ("deepseek", "oss", "qwen3"):
        or_cfg = cfg["openrouter"]
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

    # 5) Local HuggingFace model
    if name == "aryabhata":
        sec = cfg["aryabhata"]
        repo_id   = sec["model_name"]
        cache_dir = sec.get("cache_dir")
        model_path = (
            snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
            if cache_dir
            else repo_id
        )
        pipe = pipeline(
            sec.get("task", "text-generation"),
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=sec.getboolean("trust_remote_code", False),
            device_map="auto",
            max_new_tokens=sec.getint("max_new_tokens", 100),
        )
        return HuggingFacePipeline(pipeline=pipe)

    raise ValueError(f"Unsupported LLM name: {name}")
