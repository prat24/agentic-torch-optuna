# Simple Agentic Demo: AutoML using Optuna and PyTorch Lightning MCP

An **agentic demo** where an LLM autonomously calls Optuna and PyTorch Lightning MCP tools to perform simple training and hyperparameter optimization. The LLM decides which tools to call, with what parameters, and when to stop.

## Features

**LLM Tool Calling**: LLM autonomously calls tools via OpenAI function calling
**Optuna MCP**: Hyperparameter suggestion, tracking, and analysis (from PyPI)
**Lightning MCP**: Real PyTorch model training (from PyPI)

## Quick Start

```bash
# Install dependencies
uv sync

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run the demo
uv run python demo.py
```

## How It Works

1. **Connect to MCP Servers**

   - `uvx optuna-mcp` - Optuna optimization server
   - `uvx lightning-mcp` - Lightning training server
2. **Convert Tools to OpenAI Format**

   - Fetches tool schemas from both MCPs
   - Converts to OpenAI function calling format
3. **Agent Loop**

   - LLM receives available tools and task description
   - LLM calls `optuna_create_study` to start
   - LLM calls `optuna_ask` → `lightning_lightning_train` → `optuna_tell` for each trial
   - LLM calls `optuna_best_trial` to get results
   - LLM calls `finish_optimization` when done

## Project Structure

```
automl-agent/
├── demo.py          # Main agentic demo
├── pyproject.toml   # Dependencies
├── .env             # OpenAI API key (create this)
└── README.md
```

## Dependencies

- `mcp` - Model Context Protocol client
- `openai` - For GPT-4o function calling
- `python-dotenv` - Environment variables
- `optuna-mcp` - Installed via uvx at runtime
- `lightning-mcp` - Installed via uvx at runtime

## License

Apache 2.0
