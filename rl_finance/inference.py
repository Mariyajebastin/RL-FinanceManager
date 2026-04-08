import argparse
import os
import re
from typing import Any, Iterable

from pydantic import ValidationError

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = Any  # type: ignore[assignment]

try:
    from .models import RlFinanceAction
    from .server.rl_finance_environment import RlFinanceEnvironment
except ImportError:  # pragma: no cover
    from models import RlFinanceAction
    from server.rl_finance_environment import RlFinanceEnvironment


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))
load_dotenv(os.path.join(PACKAGE_DIR, ".env"))

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
BENCHMARK_NAME = "rl_finance"


def _build_client() -> OpenAI:
    if OpenAI is Any or not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _format_action(action: RlFinanceAction) -> str:
    if action.action_type == "Categorize":
        return f"Categorize(transaction_id={action.transaction_id},category={action.category})"
    if action.action_type == "FlagDuplicate":
        return f"FlagDuplicate(transaction_id={action.transaction_id})"
    if action.action_type == "SuggestCut":
        percentage = "null" if action.percentage is None else f"{action.percentage:.2f}"
        return f"SuggestCut(category={action.category},percentage={percentage})"
    return "NextPage"


def _extract_action(content: str) -> RlFinanceAction:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return RlFinanceAction.model_validate_json(match.group(0))


def _extract_action_fallback(content: str, task_name: str, observation) -> RlFinanceAction:
    text = (content or "").strip()
    lowered = text.lower()

    transaction_ids = _visible_ids(observation)
    first_visible_id = transaction_ids[0] if transaction_ids else None

    txn_match = re.search(r"TXN_\d{3}", text, re.IGNORECASE)
    transaction_id = txn_match.group(0).upper() if txn_match else first_visible_id

    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    percentage = float(pct_match.group(1)) if pct_match else 10.0

    if "nextpage" in lowered or "next page" in lowered:
        return RlFinanceAction(action_type="NextPage")

    if task_name == "medium" or "duplicate" in lowered or "flag" in lowered:
        return RlFinanceAction(action_type="FlagDuplicate", transaction_id=transaction_id)

    if task_name == "hard" or "cut" in lowered or "budget" in lowered:
        category = "Dining"
        for candidate in ("Dining", "Food", "Groceries"):
            if candidate.lower() in lowered:
                category = candidate
                break
        return RlFinanceAction(action_type="SuggestCut", category=category, percentage=percentage)

    category = "Dining"
    for candidate in (
        "Dining",
        "Income",
        "Groceries",
        "Transport",
        "Utilities",
        "Shopping",
        "Entertainment",
        "Health",
        "Housing",
        "Subscription",
    ):
        if candidate.lower() in lowered:
            category = candidate
            break
    return RlFinanceAction(
        action_type="Categorize",
        transaction_id=transaction_id,
        category=category,
    )


def _request_model_action(
    client: OpenAI | None,
    task_name: str,
    observation,
    banned_action_keys: set[str],
    banned_targets: set[str],
) -> RlFinanceAction:
    if client is None:
        raise RuntimeError("No model client configured; using local fallback policy.")

    user_prompt = _user_prompt(
        observation,
        sorted(banned_action_keys),
        sorted(banned_targets),
    )

    last_error: Exception | None = None
    request_variants = (
        {
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 80,
            "response_format": {"type": "json_object"},
        },
        {
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": user_prompt + "\nReturn only minified JSON.",
                },
            ],
            "temperature": 0,
            "max_tokens": 80,
        },
    )

    for request_kwargs in request_variants:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                **request_kwargs,
            )
            content = response.choices[0].message.content or ""
            try:
                return _extract_action(content)
            except Exception:
                return _extract_action_fallback(content, task_name, observation)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("Model request failed.")


def _system_prompt() -> str:
    return (
        "Return exactly one JSON object only.\n"
        "Valid action_type: Categorize, FlagDuplicate, SuggestCut, NextPage.\n"
        "Use transaction_id for Categorize/FlagDuplicate.\n"
        "Use category for Categorize/SuggestCut.\n"
        "Use percentage for SuggestCut.\n"
        "Never repeat a rejected action or rejected transaction/category from the banned list.\n"
        "No markdown. No prose."
    )


def _candidate_key(action: RlFinanceAction) -> str:
    parts = [action.action_type]
    if action.transaction_id:
        parts.append(action.transaction_id)
    if action.category:
        parts.append(action.category.strip().lower())
    if action.percentage is not None:
        parts.append(f"{action.percentage:.2f}")
    return "|".join(parts)


def _visible_ids(observation) -> list[str]:
    return [txn.transaction_id for txn in observation.recent_transactions]


def _infer_category(description: str, amount: float) -> str:
    lowered = _normalize_description(description)

    if amount > 0 or "salary" in lowered or "deposit" in lowered:
        return "Salary"

    category_rules = (
        ("Subscriptions", ("subscription", "netflix", "spotify", "hulu", "hbo max", "amazon prime")),
        ("Groceries", ("whole foods", "trader joe", "kroger")),
        ("Transport", ("wmata", "metro transit", "uber trip", "lyft")),
        ("Utilities", ("water", "power", "electric", "utility")),
        ("Shopping", ("target", "amazon -", "amazon ")),
        ("Health", ("cvs", "walgreens", "pharmacy", "clinic")),
        ("Housing", ("rent", "greenview")),
        ("Entertainment", ("steam", "playstation", "xbox", "cinema")),
        ("SaaS", ("digitalocean", "hosting", "server")),
        ("Dining", ("starbucks", "chipotle", "ubereats", "sweetgreen", "taco", "bistro", "steakhouse", "italian", "restaurant", "cafe")),
    )

    for category, keywords in category_rules:
        if any(keyword in lowered for keyword in keywords):
            return category

    if amount < 0:
        return "Dining"
    return "Salary"


def _normalize_description(description: str) -> str:
    return re.sub(r"\s+", " ", description.strip().lower())


def _txn_number(transaction_id: str) -> int:
    match = re.search(r"(\d+)$", transaction_id)
    return int(match.group(1)) if match else -1


def _looks_like_subscription(description: str) -> bool:
    lowered = _normalize_description(description)
    keywords = (
        "subscription",
        "netflix",
        "spotify",
        "hulu",
        "hbomax",
        "hbo max",
        "disney",
        "prime video",
        "apple music",
    )
    return any(keyword in lowered for keyword in keywords)


def _find_duplicate_candidate(
    observation,
    seen_signatures: dict[tuple[str, float], tuple[str, int]],
) -> str | None:
    for txn in observation.recent_transactions:
        signature = (_normalize_description(txn.description), round(float(txn.amount), 2))
        if signature in seen_signatures:
            prior_id, prior_num = seen_signatures[signature]
            current_num = _txn_number(txn.transaction_id)
            if _looks_like_subscription(txn.description) and 0 < (current_num - prior_num) <= 3:
                return txn.transaction_id
        seen_signatures[signature] = (txn.transaction_id, _txn_number(txn.transaction_id))
    return None


def _user_prompt(observation, banned_actions: list[str], banned_targets: list[str]) -> str:
    txns = "\n".join(
        f"{txn.transaction_id}|{txn.description}|{txn.amount:.2f}"
        for txn in observation.recent_transactions
    )
    banned_actions_str = ",".join(banned_actions[-8:]) if banned_actions else "none"
    banned_targets_str = ",".join(banned_targets[-12:]) if banned_targets else "none"
    return (
        f"task={observation.current_task_objective}\n"
        f"page={observation.current_page + 1}/{observation.total_pages}\n"
        f"failed={str(observation.last_action_failed).lower()}\n"
        f"banned_actions={banned_actions_str}\n"
        f"banned_targets={banned_targets_str}\n"
        f"visible:\n{txns if txns else 'none'}"
    )


def _fallback_action(
    task_name: str,
    observation,
    banned_targets: set[str],
    seen_signatures: dict[tuple[str, float], tuple[str, int]] | None = None,
) -> RlFinanceAction:
    visible_ids = _visible_ids(observation)

    if task_name == "medium":
        if seen_signatures is not None:
            duplicate_id = _find_duplicate_candidate(observation, seen_signatures)
            if duplicate_id and duplicate_id not in banned_targets:
                return RlFinanceAction(action_type="FlagDuplicate", transaction_id=duplicate_id)
        return RlFinanceAction(action_type="NextPage")

    if task_name == "easy":
        for txn in observation.recent_transactions:
            if txn.transaction_id not in banned_targets:
                return RlFinanceAction(
                    action_type="Categorize",
                    transaction_id=txn.transaction_id,
                    category=_infer_category(txn.description, float(txn.amount)),
                )
        return RlFinanceAction(action_type="NextPage")

    if task_name == "hard":
        for category in ("Dining", "Food", "Groceries"):
            if category.lower() not in banned_targets:
                return RlFinanceAction(
                    action_type="SuggestCut",
                    category=category,
                    percentage=10.0,
                )
        return RlFinanceAction(action_type="NextPage")

    return RlFinanceAction(action_type="NextPage")


def _normalize_action(
    task_name: str,
    action: RlFinanceAction,
    observation,
    banned_action_keys: set[str],
    banned_targets: set[str],
    seen_signatures: dict[tuple[str, float], tuple[str, int]] | None = None,
) -> RlFinanceAction:
    action_key = _candidate_key(action)
    visible_ids = set(_visible_ids(observation))

    if action_key in banned_action_keys:
        return _fallback_action(task_name, observation, banned_targets, seen_signatures)

    if action.action_type == "Categorize":
        if not action.transaction_id or action.transaction_id not in visible_ids:
            return _fallback_action(task_name, observation, banned_targets, seen_signatures)
        if action.transaction_id in banned_targets or (action.category and action.category.strip().lower() in banned_targets):
            return _fallback_action(task_name, observation, banned_targets, seen_signatures)
    elif action.action_type == "FlagDuplicate":
        if not action.transaction_id or action.transaction_id not in visible_ids or action.transaction_id in banned_targets:
            return _fallback_action(task_name, observation, banned_targets, seen_signatures)
    elif action.action_type == "SuggestCut":
        if action.percentage is None:
            action = RlFinanceAction(action_type="SuggestCut", category=action.category, percentage=10.0)
        if action.category and action.category.strip().lower() in banned_targets:
            return _fallback_action(task_name, observation, banned_targets, seen_signatures)

    return action


def _remember_failure(task_name: str, action: RlFinanceAction, banned_action_keys: set[str], banned_targets: set[str]) -> None:
    banned_action_keys.add(_candidate_key(action))

    if task_name in {"easy", "medium"} and action.transaction_id:
        banned_targets.add(action.transaction_id)
    if task_name == "easy" and action.category:
        banned_targets.add(action.category.strip().lower())
    if task_name == "hard" and action.category:
        banned_targets.add(action.category.strip().lower())


def run_episode(task_name: str, client: OpenAI | None) -> bool:
    env = RlFinanceEnvironment(task_mode=task_name)
    rewards: list[float] = []
    banned_action_keys: set[str] = set()
    banned_targets: set[str] = set()
    seen_signatures: dict[tuple[str, float], tuple[str, int]] = {}
    steps = 0
    success = False
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)
    try:
        observation = env.reset()
        done = False
        while not done:
            steps += 1
            error = None
            try:
                if client is None:
                    action = _fallback_action(task_name, observation, banned_targets, seen_signatures)
                else:
                    action = _request_model_action(
                        client,
                        task_name,
                        observation,
                        banned_action_keys,
                        banned_targets,
                    )
                action = _normalize_action(
                    task_name,
                    action,
                    observation,
                    banned_action_keys,
                    banned_targets,
                    seen_signatures,
                )
                action_str = _format_action(action)
                observation = env.step(action)
                reward = float(observation.reward or 0.0)
                done = bool(observation.done)
                error = (observation.metadata or {}).get("error")
                if reward < 0:
                    _remember_failure(task_name, action, banned_action_keys, banned_targets)
            except ValidationError as exc:
                reward = -1.0
                done = True
                action_str = "InvalidAction"
                error = exc.errors()[0]["msg"] if exc.errors() else "validation failed"
            except Exception as exc:
                action = _fallback_action(task_name, observation, banned_targets, seen_signatures)
                action = _normalize_action(
                    task_name,
                    action,
                    observation,
                    banned_action_keys,
                    banned_targets,
                    seen_signatures,
                )
                action_str = _format_action(action)
                observation = env.step(action)
                reward = float(observation.reward or 0.0)
                done = bool(observation.done)
                error = (observation.metadata or {}).get("error") or str(exc).replace("\n", " ")
                if reward < 0:
                    _remember_failure(task_name, action, banned_action_keys, banned_targets)

            rewards.append(reward)
            print(
                f"[STEP] step={steps} action={action_str} reward={reward:.2f} "
                f"done={'true' if done else 'false'} error={error if error else 'null'}",
                flush=True,
            )
            success = done and reward > 0
    finally:
        close_method = getattr(env, "close", None)
        if callable(close_method):
            close_method()
        score = max(0.0, max(rewards)) if rewards else 0.0
        rewards_str = ",".join(f"{reward:.2f}" for reward in rewards) if rewards else "0.00"
        print(
            f"[END] success={'true' if success else 'false'} steps={steps} "
            f"score={score:.2f} rewards={rewards_str}",
            flush=True,
        )
    return success


def run_inference(task_mode: str | None = None) -> list[tuple[str, bool]]:
    requested_mode = (task_mode or os.getenv("TASK_MODE", "random")).strip().lower()
    modes: Iterable[str]
    if requested_mode == "all":
        modes = ("easy", "medium", "hard")
    elif requested_mode in {"easy", "medium", "hard", "random"}:
        modes = (requested_mode,)
    else:
        raise ValueError("task mode must be one of: easy, medium, hard, random, all")

    client = _build_client()
    return [(mode, run_episode(mode, client)) for mode in modes]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-mode",
        choices=["random", "easy", "medium", "hard", "all"],
        default=None,
        help="Choose which task mode to run. Defaults to TASK_MODE env var or random.",
    )
    args = parser.parse_args(argv)
    try:
        run_inference(task_mode=args.task_mode)
    except Exception as exc:
        print(f"[STEP] step=0 action=StartupError reward=0.00 done=true error={str(exc).replace(chr(10), ' ')}", flush=True)
        print("[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 
 
