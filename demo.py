#!/usr/bin/env python3
"""
Agentic AutoML Demo - LLM with MCP Tool Calling

A demonstration of an LLM agent that autonomously calls MCP tools to perform
hyperparameter optimization. The agent discovers available tools from Optuna MCP
and Lightning MCP, then decides which tools to call, with what parameters, and
when to stop.

Usage:
    uv run python demo.py

Requirements:
    - OPENAI_API_KEY in .env file
    - optuna-mcp and lightning-mcp (installed via uvx at runtime)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import OpenAI

# Configuration
MODEL = "gpt-4o"
MAX_ITERATIONS = 25
RATE_LIMIT_RETRY_DELAY = 10

# Setup
load_dotenv()
logging.basicConfig(level=logging.WARNING)


def get_openai_client() -> OpenAI:
    """Initialize and return OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return OpenAI(api_key=api_key)


def extract_result(call_result: Any) -> dict[str, Any] | str:
    """Extract text or JSON from MCP tool result.

    Args:
        call_result: Raw result from MCP tool call.

    Returns:
        Parsed JSON dict or raw text string.
    """
    if hasattr(call_result, "content") and call_result.content:
        for block in call_result.content:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)
                except (json.JSONDecodeError, TypeError):
                    return block.text
    return {}


# Tool description templates
OPTUNA_ASK_DESC = """Get next hyperparameter suggestions from Optuna. REQUIRED: search_space parameter.

Example: {"search_space": {"input_dim": {"name": "IntDistribution", "attributes": {"low": 4, "high": 16}}, "num_classes": {"name": "IntDistribution", "attributes": {"low": 2, "high": 5}}, "lr": {"name": "FloatDistribution", "attributes": {"low": 0.001, "high": 0.1, "log": true}}}}"""

LIGHTNING_TRAIN_DESC = """Train a PyTorch Lightning model. REQUIRED: model and trainer configs.

Example: {"model": {"_target_": "lightning_mcp.models.simple.SimpleClassifier", "input_dim": 10, "num_classes": 3, "lr": 0.01}, "trainer": {"max_epochs": 5, "accelerator": "cpu", "enable_progress_bar": false, "enable_model_summary": false}}"""


def mcp_tools_to_openai_tools(optuna_tools: list, lightning_tools: list) -> list[dict]:
    """Convert MCP tool schemas to OpenAI function calling format.

    Args:
        optuna_tools: List of tools from Optuna MCP server.
        lightning_tools: List of tools from Lightning MCP server.

    Returns:
        List of tools in OpenAI function calling format.
    """
    openai_tools = []

    # Add Optuna tools
    for tool in optuna_tools:
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        desc = OPTUNA_ASK_DESC if tool.name == "ask" else (tool.description[:500] if tool.description else "Optuna tool")

        openai_tools.append({
            "type": "function",
            "function": {
                "name": f"optuna_{tool.name}",
                "description": desc,
                "parameters": schema,
            },
        })

    # Add Lightning tools
    for tool in lightning_tools:
        name = tool.name.replace(".", "_")  # OpenAI doesn't allow dots in function names
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        desc = LIGHTNING_TRAIN_DESC if "train" in tool.name else (tool.description[:500] if tool.description else "Lightning tool")

        openai_tools.append({
            "type": "function",
            "function": {
                "name": f"lightning_{name}",
                "description": desc,
                "parameters": schema,
            },
        })
    
    # Add finish tool for agent to signal completion
    openai_tools.append({
        "type": "function",
        "function": {
            "name": "finish_optimization",
            "description": "Call when optimization is complete. MUST include best_params.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of optimization results",
                    },
                    "best_params": {
                        "type": "object",
                        "description": "Best hyperparameters: {input_dim, num_classes, lr}",
                        "properties": {
                            "input_dim": {"type": "integer"},
                            "num_classes": {"type": "integer"},
                            "lr": {"type": "number"},
                        },
                        "required": ["input_dim", "num_classes", "lr"],
                    },
                    "best_loss": {
                        "type": "number",
                        "description": "Best training loss achieved",
                    },
                },
                "required": ["summary", "best_params", "best_loss"],
            },
        },
    })

    return openai_tools


async def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    optuna_sess: ClientSession,
    lightning_sess: ClientSession,
) -> str:
    """Execute an MCP tool and return the result as JSON string.

    Args:
        tool_name: Name of the tool to execute (with prefix).
        args: Arguments to pass to the tool.
        optuna_sess: Optuna MCP client session.
        lightning_sess: Lightning MCP client session.

    Returns:
        JSON string with the tool result.
    """
    if tool_name == "finish_optimization":
        return json.dumps({"status": "optimization_complete", **args})

    if tool_name.startswith("optuna_"):
        mcp_tool_name = tool_name.removeprefix("optuna_")
        result = await optuna_sess.call_tool(mcp_tool_name, args)
        return json.dumps(extract_result(result))

    if tool_name.startswith("lightning_"):
        mcp_tool_name = tool_name.removeprefix("lightning_").replace("_", ".")
        result = await lightning_sess.call_tool(mcp_tool_name, args)
        return json.dumps(extract_result(result))

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# System prompt for the agent
SYSTEM_PROMPT = """You are an AutoML agent that optimizes hyperparameters using MCP tools.

You have access to:
1. Optuna MCP tools (optuna_*) - for managing optimization studies
2. Lightning MCP tools (lightning_*) - for training PyTorch models

Your task (follow these steps IN ORDER):
1. Create an Optuna study: optuna_create_study(study_name="automl", directions=["minimize"])
2. Run EXACTLY 3 training trials. For each trial:
   a) Call optuna_ask with search_space to get hyperparameters
   b) Call lightning_lightning_train to train the model
   c) Call optuna_tell to report the train_loss
3. After 3 trials, call optuna_best_trial to get the best result
4. IMMEDIATELY call finish_optimization with the summary

IMPORTANT: After 3 trials, you MUST call finish_optimization.

For lightning_lightning_train:
{"model": {"_target_": "lightning_mcp.models.simple.SimpleClassifier", "input_dim": <int>, "num_classes": <int>, "lr": <float>}, "trainer": {"max_epochs": 5, "accelerator": "cpu", "enable_progress_bar": false, "enable_model_summary": false}}

For optuna_ask:
{"search_space": {"input_dim": {"name": "IntDistribution", "attributes": {"low": 4, "high": 16}}, "num_classes": {"name": "IntDistribution", "attributes": {"low": 2, "high": 5}}, "lr": {"name": "FloatDistribution", "attributes": {"low": 0.001, "high": 0.1, "log": true}}}}

Call tools one at a time. Count your trials carefully."""


async def main() -> None:
    """Run the agentic AutoML demo."""
    print("\n" + "=" * 70)
    print("🤖 AGENTIC AUTOML DEMO - LLM Tool Calling")
    print("=" * 70)
    print("\nThe LLM agent will autonomously:")
    print("   • Discover available MCP tools")
    print("   • Decide which tools to call")
    print("   • Run hyperparameter optimization")
    print("   • Analyze results and iterate")
    print("=" * 70 + "\n")

    client = get_openai_client()

    # MCP server configurations
    optuna_params = StdioServerParameters(command="uvx", args=["optuna-mcp"])
    lightning_params = StdioServerParameters(command="uvx", args=["lightning-mcp"])

    print("🔌 Connecting to MCP servers...")
    
    async with stdio_client(optuna_params) as optuna_io:
        async with ClientSession(optuna_io[0], optuna_io[1]) as optuna_sess:
            await optuna_sess.initialize()
            print("   ✅ Connected to Optuna MCP")

            async with stdio_client(lightning_params) as lightning_io:
                async with ClientSession(lightning_io[0], lightning_io[1]) as lightning_sess:
                    await lightning_sess.initialize()
                    print("   ✅ Connected to Lightning MCP")

                    # Discover available tools
                    optuna_tools_list = await optuna_sess.list_tools()
                    lightning_tools_list = await lightning_sess.list_tools()

                    print(f"\n📋 Available Tools:")
                    print(f"   Optuna MCP: {len(optuna_tools_list.tools)} tools")
                    print(f"   Lightning MCP: {len(lightning_tools_list.tools)} tools")

                    # Convert MCP tools to OpenAI format
                    openai_tools = mcp_tools_to_openai_tools(
                        optuna_tools_list.tools,
                        lightning_tools_list.tools,
                    )

                    # Initialize conversation
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": "Run hyperparameter optimization for a classifier using Optuna and Lightning MCP tools.",
                        },
                    ]

                    print("\n" + "=" * 70)
                    print("🧠 AGENT STARTING - LLM will autonomously call tools")
                    print("=" * 70)

                    # Agent loop
                    await run_agent_loop(
                        client=client,
                        messages=messages,
                        openai_tools=openai_tools,
                        optuna_sess=optuna_sess,
                        lightning_sess=lightning_sess,
                    )
                    

async def run_agent_loop(
    client: OpenAI,
    messages: list[dict],
    openai_tools: list[dict],
    optuna_sess: ClientSession,
    lightning_sess: ClientSession,
) -> None:
    """Run the agent loop until completion or max iterations.

    Args:
        client: OpenAI client instance.
        messages: Conversation message history.
        openai_tools: Tools in OpenAI format.
        optuna_sess: Optuna MCP client session.
        lightning_sess: Lightning MCP client session.
    """
    for iteration in range(MAX_ITERATIONS):
        # Call LLM with retry for rate limits
        response = await call_llm_with_retry(client, messages, openai_tools)
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        # Process tool calls
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                print(f"\n🔧 Agent calls: {tool_name}")
                print(f"   Args: {json.dumps(tool_args, indent=2)[:200]}")

                result = await execute_tool(tool_name, tool_args, optuna_sess, lightning_sess)
                print(f"   Result: {result[:300]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

                # Check for completion
                if tool_name == "finish_optimization":
                    print_final_results(json.loads(result))
                    return

        # Agent thinking (no tool calls)
        elif assistant_message.content:
            print(f"\n💭 Agent: {assistant_message.content[:200]}...")

        # Check for unexpected stop
        if response.choices[0].finish_reason == "stop" and not assistant_message.tool_calls:
            print("\n⚠️ Agent stopped without calling finish_optimization")
            return

    print(f"\n⚠️ Reached max iterations ({MAX_ITERATIONS})")


async def call_llm_with_retry(client: OpenAI, messages: list[dict], tools: list[dict]):
    """Call LLM with retry logic for rate limits."""
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
            )
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < 2:
                print(f"   ⏳ Rate limited, waiting {RATE_LIMIT_RETRY_DELAY}s...")
                time.sleep(RATE_LIMIT_RETRY_DELAY)
            else:
                raise


def print_final_results(result: dict) -> None:
    """Print the final optimization results."""
    print("\n" + "=" * 70)
    print("🏆 AGENT FINISHED OPTIMIZATION")
    print("=" * 70)
    print(f"\n{result.get('summary', 'No summary')}")
    print(f"\nBest params: {result.get('best_params', {})}")
    print(f"Best loss: {result.get('best_loss', 'N/A')}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
