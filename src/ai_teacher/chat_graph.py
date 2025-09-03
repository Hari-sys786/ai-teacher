from typing import Dict, Any
import pathlib
import re

from langgraph.graph import StateGraph, END
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import Runnable
from transformers import TextGenerationPipeline, GenerationConfig

from llm_selector import get_llm


# ────────────────────────────────────────────────────────────
# 1) Session-memory
# ────────────────────────────────────────────────────────────
memory = ConversationBufferMemory(return_messages=True)


# ────────────────────────────────────────────────────────────
# 2) Aryabhata wrapper that
#    • uses the official chat-template
#    • is callable ( __call__ = invoke )
#    • stops ONLY at <|im_end|>  (never mid sentence)
# ────────────────────────────────────────────────────────────
class AryRunnable(Runnable):
    def __init__(self, pipe: TextGenerationPipeline):
        self.pipe = pipe
        cfg = GenerationConfig.from_pretrained(
            pathlib.Path(pipe.model.config.name_or_path)
        )
        self.gen_kw = dict(
            temperature=0.2,
            top_p=0.7,
            max_new_tokens=256,                     # plenty of room
            return_full_text=False,
            eos_token_id=cfg.eos_token_id,
            pad_token_id=cfg.eos_token_id,
        )
        self.stop_re = re.compile(r"<\|im_end\|>|<\|end\|>")

    # make the object callable
    def invoke(self, prompt: str, _=None) -> str:
        txt = self.pipe(prompt, **self.gen_kw)[0]["generated_text"]
        return self.stop_re.split(txt)[0].strip()

    __call__ = invoke


# ────────────────────────────────────────────────────────────
# 3) Build the LangGraph
# ────────────────────────────────────────────────────────────
def _build_graph():
    ary_llm = AryRunnable(get_llm("aryabhata"))
    sg = StateGraph(dict)

    # helper: render last N messages cleanly
    def last_n_render(n: int):
        msgs = memory.load_memory_variables({})["history"][-n:]
        out = []
        for m in msgs:
            role = "user" if m.type == "human" else "assistant"
            out.append({"role": role, "content": m.content})
        return out

    def ary_node(state: Dict[str, Any]):
        user = state["user"]
        memory.chat_memory.add_user_message(user)

        # build chat prompt (history + current user)
        chat_list = last_n_render(6)        # ← only 6 recent messages
        chat_list.append({"role": "user", "content": user})

        prompt = ary_llm.pipe.tokenizer.apply_chat_template(
            chat_list,
            tokenize=False,
            add_generation_prompt=True,
        )

        reply = ary_llm(prompt)
        memory.chat_memory.add_ai_message(reply)
        state["ary"] = reply
        return state

    sg.add_node("ary", ary_node)
    sg.set_entry_point("ary")
    sg.add_edge("ary", END)
    return sg.compile()


GRAPH = _build_graph()
