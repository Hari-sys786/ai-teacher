# AI Teacher

AI Teacher is an interactive CLI and HTTP API that lets you swap between multiple LLM backends—OpenAI, Google Gemini, Ollama, OpenRouter-hosted free models, or local Hugging Face models (e.g. PhysicsWallahAI/Aryabhata-1.0)—with a single unified interface.

#### Prerequisites  
• Python 3.11 or later  
• Git (to clone the repo)  
• [Poetry](https://python-poetry.org/) for dependency management  

#### Installation  

1. Install Poetry (if you don’t already have it):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   export PATH="$HOME/.local/bin:$PATH"
   poetry --version
   ```

2. Clone this repository and enter its folder:

   ```bash
   git clone https://github.com/your-username/ai-teacher.git
   cd ai-teacher
   ```

3. Install all dependencies into a new virtual environment:

   ```bash
   poetry install
   ```

#### Configuration  

Edit the `config.ini` at the project root to choose your default model and supply credentials:

```ini
[llm]
choice = deepseek

[openrouter]
api_key  = sk-your_openrouter_key
base_url = https://openrouter.ai/api/v1

[deepseek]
provider   = openrouter
model_name = deepseek/deepseek-r1-0528-qwen3-8b:free

[oss]
provider   = openrouter
model_name = openai/gpt-oss-120b:free

[qwen3]
provider   = openrouter
model_name = qwen/qwen3-coder:free

[aryabhata]
provider           = huggingface
model_name         = PhysicsWallahAI/Aryabhata-1.0
cache_dir          = ./models/aryabhata
trust_remote_code  = true
task               = text-generation
max_new_tokens     = 256
```

#### (Optional) Pre-download the Aryabhata Model  

```bash
poetry run python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="PhysicsWallahAI/Aryabhata-1.0",
  cache_dir="./models/aryabhata"
)
EOF
```

This ensures the model weights and custom code are cached locally before you start.

#### CLI Usage  

Run the interactive prompt:

```bash
poetry run python -m ai_teacher.main
```

You’ll see:

```
🤖 AI Teacher (type 'end', 'stop' or 'exit' to quit)
You:
```

Type your question and press Enter. The AI’s answer will be printed. Type `end`, `stop`, or `exit` to quit.

#### HTTP API Usage  

Start the FastAPI server:

```bash
poetry run uvicorn ai_teacher.api:app --reload --host 0.0.0.0 --port 8000
```

Send a POST request to `/teach`:

```bash
curl http://localhost:8000/teach \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain Gauss’s law."}'
```

#### Extending  

• To add a new model, simply add a `[newmodel]` section in `config.ini` with `provider = openrouter`, `provider = huggingface`, or `provider = local`.  
• Call it by name in `get_llm("newmodel")` or update your CLI/API code as needed.

#### License  

MIT License. Feel free to fork, modify, and use as you like!