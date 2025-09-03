# from llm_selector import get_llm
# from transformers import GenerationConfig, TextGenerationPipeline
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import Runnable
# import re, pathlib


# # ── 1. pipeline + config ───────────────────────────────────────────
# pipe      = get_llm("aryabhata")
# repo_path = pathlib.Path(pipe.model.config.name_or_path)
# gen_cfg   = GenerationConfig.from_pretrained(repo_path)

# STOP_RE = re.compile(r"<\|im_end\|>|<\|end\|>")
# BOX_RE  = re.compile(r"\\boxed\{([^}]*)\}")

# # ── 2. wrapper that yields both pieces ─────────────────────────────
# class AryExplainAndAnswer(Runnable):
#     def __init__(self, pipeline):
#         self.pipe = pipeline
#         self.kw   = dict(
#             temperature=0.2,
#             top_p=0.7,
#             max_new_tokens=gen_cfg.max_new_tokens or 192,
#             return_full_text=False,
#             eos_token_id=gen_cfg.eos_token_id,
#             pad_token_id=gen_cfg.eos_token_id,
#         )

#     def invoke(self, inp, config=None) -> str:
#         user = inp["query"] if isinstance(inp, dict) else str(inp)

#         prompt = self.pipe.tokenizer.apply_chat_template(
#             [
#                 {"role": "system",
#                  "content": "Think step-by-step; put ONLY the final answer inside \\boxed{ }."},
#                 {"role": "user", "content": user},
#             ],
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#         text = self.pipe(prompt, **self.kw)[0]["generated_text"]
#         text = STOP_RE.split(text)[0].strip()

#         # try to isolate the boxed result
#         m = BOX_RE.search(text)
#         if m:
#             final = m.group(1).strip()
#             return f"{text}\n\nFinal answer: {final}"
#         return text
    

# def build_chain(model_choice: str):
#     llm = get_llm(model_choice)

#     # If it's a HF pipeline → wrap; else use as-is
#     if isinstance(llm, TextGenerationPipeline):
#         runnable_llm = AryExplainAndAnswer(llm)
#     else:
#         # Chat models already follow LangChain Runnable spec
#         system_msg = (
#             "Think step-by-step; "
#             "put ONLY the final answer inside."
#         )
#         def chat_invoke(inp, _=None):
#             q = inp["query"] if isinstance(inp, dict) else str(inp)
#             prompt = f"{system_msg}\n\nQuestion: {q}"
#             # ChatOpenAI / Gemini / etc. expose .invoke or __call__
#             return (llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt))

#         class ChatRunnable(Runnable):
#             invoke = staticmethod(chat_invoke)

#         runnable_llm = ChatRunnable()

#     prompt = PromptTemplate.from_template("{query}")
#     return prompt | runnable_llm | StrOutputParser()


# # # ── 3. LangChain chain ─────────────────────────────────────────────
# # prompt = PromptTemplate.from_template("{query}")
# # chain  = prompt | AryExplainAndAnswer(pipe) | StrOutputParser()


# def main():
#     print("🤖 Aryabhata Tutor (type 'exit' to quit)")
#     chain = build_chain("qwen3")
#     while True:
#         q = input("You: ").strip()
#         if q.lower() in {"exit", "stop", "end"}:
#             break
#         try:
#             print("AI:", chain.invoke({"query": q}))
#         except Exception as e:
#             print("⚠️", e)


# if __name__ == "__main__":
#     main()
from chat_graph import GRAPH

def main() -> None:
    print("🤖 Aryabhata (type 'exit' to quit)")
    while True:
        text = input("You: ").strip()
        if text.lower() in {"exit", "stop", "end"}:
            break
        try:
            state  = GRAPH.invoke({"user": text})   # ← key must be 'user'
            print("AI:", state["ary"])              # reply lives under 'ary'
        except Exception as err:
            print("⚠️", err)

if __name__ == "__main__":
    main()
