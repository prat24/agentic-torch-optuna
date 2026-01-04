#!/usr/bin/env python3
"""
Agentic AutoML Demo - LLM autonomously calls MCP tools for hyperparameter
optimization with full ML workflow: train, validate, test, and checkpoint.

Usage:
    uv run python demo.py

Requirements:
    - OPENAI_API_KEY in .env file
    - optuna-mcp and lightning-mcp>=0.5.0 (installed via uvx at runtime)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import OpenAI

# Configuration
MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 30
RATE_LIMIT_RETRY_DELAY = 3

# Setup
load_dotenv()
logging.basicConfig(level=logging.WARNING)


@dataclass
class MCPConnection:
    """Holds MCP session and metadata."""

    session: ClientSession
    prefix: str
    tools: list[Any]


def get_openai_client() -> OpenAI:
    """Create OpenAI client from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return OpenAI(api_key=api_key)


def extract_result(call_result: Any) -> dict[str, Any] | str:
    """Extract JSON or text from an MCP tool result."""
    if hasattr(call_result, "content") and call_result.content:
        for block in call_result.content:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    return block.text
    return {}


def mcp_tool_to_openai_function(tool: Any, prefix: str) -> dict[str, Any]:
    """Convert a single MCP tool to OpenAI function format."""
    # Strip prefix from tool name if present (e.g., lightning.train -> train)
    tool_name = tool.name
    if tool_name.startswith(f"{prefix}."):
        tool_name = tool_name[len(prefix) + 1 :]

    # Normalize: replace dots with underscores for OpenAI compatibility
    normalized_name = tool_name.replace(".", "_")
    full_name = f"{prefix}_{normalized_name}"

    description = tool.description or f"{prefix} tool: {tool.name}"

    # Add examples for tools that need them (helps smaller models)
    if tool.name == "ask":
        description = 'Get next hyperparameter suggestions. The search_space parameter defines the hyperparameter distributions to sample from.'

    return {
        "type": "function",
        "function": {
            "name": full_name,
            "description": description,
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def build_openai_tools(connections: list[MCPConnection]) -> list[dict[str, Any]]:
    """Build OpenAI tools from all MCP connections plus finish sentinel."""
    openai_tools = []

    for conn in connections:
        for tool in conn.tools:
            openai_tools.append(mcp_tool_to_openai_function(tool, conn.prefix))

    # Sentinel tool to signal completion
    openai_tools.append({
        "type": "function",
        "function": {
            "name": "finish_optimization",
            "description": "Signal that the optimization workflow is complete. Call this after testing the best model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of the optimization process and results",
                    },
                    "best_params": {
                        "type": "object",
                        "description": "The best hyperparameters found",
                    },
                    "best_metrics": {
                        "type": "object",
                        "description": "Metrics from the best trial (train_loss, val_loss, test_loss, etc.)",
                    },
                    "checkpoint_path": {
                        "type": "string",
                        "description": "Path to the saved checkpoint of the best model (if saved)",
                    },
                },
                "required": ["summary", "best_params", "best_metrics"],
            },
        },
    })

    return openai_tools


async def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    connections: dict[str, MCPConnection],
) -> str:
    """Route and execute a tool call to the appropriate MCP server."""
    if tool_name == "finish_optimization":
        return json.dumps({"status": "optimization_complete", **args})

    # Find the matching connection by prefix
    for prefix, conn in connections.items():
        if tool_name.startswith(f"{prefix}_"):
            # Convert back to MCP tool name
            # lightning_train -> lightning.train
            # optuna_create_study -> create_study
            suffix = tool_name.removeprefix(f"{prefix}_")
            if prefix == "lightning":
                # Lightning tools: train -> lightning.train
                mcp_tool_name = f"lightning.{suffix}"
            else:
                # Optuna tools use underscores as-is
                mcp_tool_name = suffix

            try:
                result = await conn.session.call_tool(mcp_tool_name, args)
                return json.dumps(extract_result(result))
            except Exception as e:
                return json.dumps({"error": str(e), "tool": mcp_tool_name, "args": args})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# System prompt - truly agentic, tool-driven
SYSTEM_PROMPT = """You are an AutoML agent with access to Optuna (hyperparameter optimization) and Lightning (ML training) tools.

YOUR GOAL: Find the best hyperparameters for a neural network classifier, then save and test it.

AVAILABLE TOOLS:
- optuna_create_study: Create an optimization study
- optuna_ask: Sample hyperparameters from a search space  
- optuna_tell: Report the objective value for a trial
- optuna_best_trial: Get the best trial's parameters
- lightning_train: Train a model
- lightning_validate: Validate a trained model (gets val_loss)
- lightning_test: Test a trained model (gets test metrics)
- lightning_checkpoint: Save/load model checkpoints
- finish_optimization: Call when done

MODEL TO OPTIMIZE: model.DemoClassifier
- Required params: input_dim (int), num_classes (int), lr (float)
- Use _target_: "model.DemoClassifier" in model config

SEARCH SPACE FORMAT (for optuna_ask):
{
  "param_name": {
    "name": "IntDistribution" | "FloatDistribution",
    "attributes": {"low": X, "high": Y, "log": true/false}
  }
}

TRAINER CONFIG:
{"max_epochs": 5, "accelerator": "cpu", "enable_progress_bar": false, "enable_model_summary": false}

WORKFLOW PATTERN:
1. Create study (minimize val_loss)
2. For each trial: ask → train → validate → tell (with val_loss from validate)
3. After trials: get best_trial → train best → checkpoint → test → finish

RULES:
- ONE tool call per turn
- Read ACTUAL values from tool responses, don't assume
- Use val_loss from lightning_validate result for optuna_tell
- trial_number comes from optuna_ask response"""


async def run_agent_loop(
    client: OpenAI,
    messages: list[dict],
    openai_tools: list[dict],
    connections: dict[str, MCPConnection],
) -> dict[str, Any] | None:
    """Run the agent loop until completion or max iterations."""
    for iteration in range(MAX_ITERATIONS):
        response = call_llm_with_retry(client, messages, openai_tools)
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = (
                    json.loads(tool_call.function.arguments)
                    if tool_call.function.arguments
                    else {}
                )

                print(f"\n[{iteration + 1}] {tool_name}")
                print(f"    Args: {json.dumps(tool_args, indent=2)[:300]}")

                result = await execute_tool(tool_name, tool_args, connections)
                result_preview = result[:400] + "..." if len(result) > 400 else result
                print(f"    Result: {result_preview}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

                if tool_name == "finish_optimization":
                    return json.loads(result)

        elif assistant_message.content:
            print(f"\nAgent: {assistant_message.content[:300]}...")

        # Check for natural stop without finish
        if response.choices[0].finish_reason == "stop" and not assistant_message.tool_calls:
            print("\n⚠️  Agent stopped without calling finish_optimization")
            return None

    print(f"\n⚠️  Reached max iterations ({MAX_ITERATIONS})")
    return None


def call_llm_with_retry(
    client: OpenAI, messages: list[dict], tools: list[dict], max_retries: int = 3
) -> Any:
    """Call the LLM with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                print(f"Rate limited, waiting {RATE_LIMIT_RETRY_DELAY}s...")
                time.sleep(RATE_LIMIT_RETRY_DELAY)
            else:
                raise


def print_banner() -> None:
    """Print startup banner."""
    print("\n" + "=" * 70)
    print("AGENTIC AUTOML DEMO")
    print("=" * 70)
    print("\nWorkflow: Optimize → Train → Validate → Checkpoint → Test")
    print("\nThe agent will autonomously:")
    print("  • Discover available MCP tools")
    print("  • Run hyperparameter optimization trials")
    print("  • Validate model during optimization")
    print("  • Save checkpoint of best model")
    print("  • Run final test evaluation")
    print("=" * 70)


def print_results(result: dict[str, Any]) -> None:
    """Print final optimization results."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\n{result.get('summary', 'No summary provided')}")
    print(f"\nBest Parameters: {json.dumps(result.get('best_params', {}), indent=2)}")
    print(f"\nMetrics: {json.dumps(result.get('best_metrics', {}), indent=2)}")
    if result.get("checkpoint_path"):
        print(f"\nCheckpoint: {result.get('checkpoint_path')}")
    print("\n" + "=" * 70)


async def main() -> None:
    """Run the agentic AutoML demo."""
    print_banner()

    client = get_openai_client()

    # MCP server configurations
    # Optuna: use uvx (isolated env is fine)
    optuna_params = StdioServerParameters(command="uvx", args=["optuna-mcp"])
    # Lightning: run via CLI entry point which properly redirects stderr
    # Use local venv so it can import our model.py
    lightning_params = StdioServerParameters(
        command="uv",
        args=["run", "lightning-mcp"],
    )

    print("\nConnecting to MCP servers...")

    async with stdio_client(optuna_params) as optuna_io:
        async with ClientSession(optuna_io[0], optuna_io[1]) as optuna_sess:
            await optuna_sess.initialize()
            optuna_tools = await optuna_sess.list_tools()
            print(f"✓ Optuna MCP: {len(optuna_tools.tools)} tools")

            async with stdio_client(lightning_params) as lightning_io:
                async with ClientSession(lightning_io[0], lightning_io[1]) as lightning_sess:
                    await lightning_sess.initialize()
                    lightning_tools = await lightning_sess.list_tools()
                    print(f"✓ Lightning MCP: {len(lightning_tools.tools)} tools")

                    # List discovered tools
                    print("\nLightning tools available:")
                    for tool in lightning_tools.tools:
                        print(f"  • {tool.name}: {tool.description[:60]}...")

                    # Build connections map
                    connections = {
                        "optuna": MCPConnection(
                            session=optuna_sess,
                            prefix="optuna",
                            tools=optuna_tools.tools,
                        ),
                        "lightning": MCPConnection(
                            session=lightning_sess,
                            prefix="lightning",
                            tools=lightning_tools.tools,
                        ),
                    }

                    # Build OpenAI tools from discovered MCP tools
                    openai_tools = build_openai_tools(list(connections.values()))

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": "Optimize hyperparameters for DemoClassifier using 3 trials. Search space: input_dim (4-16), num_classes (2-5), lr (0.001-0.1 log scale). After finding the best config, save a checkpoint and run test evaluation. Report final results.",
                        },
                    ]

                    print("\n" + "=" * 70)
                    print("AGENT STARTING")
                    print("=" * 70)

                    result = await run_agent_loop(
                        client=client,
                        messages=messages,
                        openai_tools=openai_tools,
                        connections=connections,
                    )

                    if result:
                        print_results(result)
                    else:
                        print("\n❌ Optimization did not complete successfully")


if __name__ == "__main__":
    asyncio.run(main())
