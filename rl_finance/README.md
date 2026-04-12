---
title: RL Finance Manager
emoji: "\U0001F4B8"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# RL Finance Manager

RL Finance Manager is an OpenEnv-compatible environment for evaluating agents on practical personal-finance workflows. The agent sees a masked transaction history and must complete one of three tasks: categorize a transaction, flag a duplicate subscription charge, or recommend a budget cut.

## Environment Description

This environment simulates a realistic financial-review workflow instead of a toy task. The dataset contains mock transactions across income, groceries, dining, subscriptions, utilities, housing, transport, and discretionary spending. Hidden labels stay inside the environment so the agent is graded only through its actions.

## Action Space

The environment accepts one structured action per step through `RlFinanceAction`.

- `Categorize`: provide `transaction_id` and `category`
- `FlagDuplicate`: provide `transaction_id`
- `SuggestCut`: provide `category` and `percentage`
- `NextPage`: move to the next page of transactions

## Observation Space

Each step returns an `RlFinanceObservation` with:

- `current_balance`
- `recent_transactions`
- `current_task_objective`
- `last_action_failed`
- `current_page`
- `total_pages`
- `total_transactions`

Transactions are masked views containing:

- `transaction_id`
- `date`
- `amount`
- `description`

## Tasks

| Task | Difficulty | Objective | Success signal |
| --- | --- | --- | --- |
| `easy` | Easy | Categorize a transaction correctly | `+0.10` for the correct category |
| `medium` | Medium | Flag the duplicate subscription charge | `+1.00` for the correct duplicate |
| `hard` | Hard | Recommend cutting dining spend by `10%` | `+1.00` for the correct category and percentage |

## Reward Design

The reward function gives partial progress signals and discourages unproductive behavior:

- correct categorization gives a small positive reward
- duplicate detection and correct budget cuts give full-task rewards
- invalid or incorrect actions receive penalties
- pagination carries a small penalty to discourage aimless scrolling
- trying to page past the end receives a larger penalty

Episodes end on task completion or when the step budget is exhausted.

## Local Setup

From the repository root:

```bash
cd rl_finance
uv sync
uv pip install -r requirements.txt
```

Run the server:

```bash
cd rl_finance
uv run server --port 8000
```

## Baseline Inference

The submission baseline script lives at the repository root as `inference.py`.

That root-level wrapper is intentionally hardened so that import or startup failures still emit a minimal `[START]`, `[STEP]`, and `[END]` sequence to stdout, which makes grader-visible failures easier to diagnose.

Set your inference environment:

```bash
export API_KEY="your_api_key_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b"
```

Run one task:

```bash
python inference.py --task-mode easy
```

By default, `python inference.py` runs the `easy` task so stdout contains one clean episode for validator-friendly parsing.

Run all required tasks:

```bash
python inference.py --task-mode all
```

The script uses the OpenAI Python client and emits the required `[START]`, `[STEP]`, and `[END]` lines for each episode.

## Baseline Scores

Run `python inference.py --task-mode all` from the repository root after setting `API_KEY`, then replace the placeholders below with the recorded scores:

| Task | Model | Score |
| --- | --- | --- |
| `easy` | `openai/gpt-oss-120b` | `0.10` |
| `medium` | `openai/gpt-oss-120b` | `1.00` |
| `hard` | `openai/gpt-oss-120b` | `1.00` |

## Docker

Build locally from the environment directory:

```bash
cd rl_finance
docker build -t rl-finance-env .
```

Run locally:

```bash
docker run --rm -p 8000:8000 rl-finance-env
```

## Hugging Face Spaces

This environment is packaged for a Docker-based Hugging Face Space and includes the `openenv` tag in the front matter above. Once you are authenticated with Hugging Face, deploy from the `rl_finance` directory:

```bash
env PATH="/home/redark/.local/bin:$PATH" hf auth login
env PATH="/home/redark/.local/bin:$PATH" hf auth whoami
./.venv/bin/openenv push --repo-id YOUR_USERNAME/rl-finance-manager
```
