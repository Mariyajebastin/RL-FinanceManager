import argparse
import os
import random
import re
from typing import Any, Iterable

try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover
    class ValidationError(Exception):
        pass

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
    from . import RlFinanceEnv
    from .server.rl_finance_environment import RlFinanceEnvironment
except ImportError:  # pragma: no cover
    from models import RlFinanceAction
    try:
        from __init__ import RlFinanceEnv
    except ImportError:  # pragma: no cover
        RlFinanceEnv = None
    from server.rl_finance_environment import RlFinanceEnvironment


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))
load_dotenv(os.path.join(PACKAGE_DIR, ".env"))

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK_NAME = "rl_finance"
SUPPORTED_TASK_MODES = frozenset({"easy", "medium", "hard", "random", "all"})


class StructuredArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _safe_text(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return text or value.__class__.__name__


def _startup_task_label(task_name: str | None) -> str:
    candidate = (task_name or "").strip().lower()
    return candidate if candidate in SUPPORTED_TASK_MODES else "startup"


def _emit_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)


def _emit_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={error if error else 'null'}",
        flush=True,
    )


def _emit_end(success: bool, steps: int, rewards: Iterable[float]) -> None:
    rewards_list = [float(reward) for reward in rewards]
    score = max(0.0, max(rewards_list)) if rewards_list else 0.0
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards_list) if rewards_list else "0.00"
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _emit_startup_failure(exc: BaseException, task_name: str | None = None) -> int:
    _emit_start(_startup_task_label(task_name))
    _emit_step(
        step=0,
        action="StartupError",
        reward=0.0,
        done=True,
        error=_safe_text(exc),
    )
    _emit_end(success=False, steps=0, rewards=[0.0])
    return 0


def _task_mode_from_unknown_args(argv: Iterable[str]) -> str | None:
    for raw_arg in argv:
        token = raw_arg.strip().lower()
        if token in SUPPORTED_TASK_MODES:
            return token
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.lstrip("-") in {"task", "task_name", "task-name", "task_mode", "task-mode", "mode"}:
            if value in SUPPORTED_TASK_MODES:
                return value
    return None


def _required_env(name: str) -> str:
    try:
        value = os.environ[name].strip()
    except KeyError as exc:
        raise RuntimeError(f"{name} is required for grader proxy calls.") from exc
    if not value:
        raise RuntimeError(f"{name} must not be empty.")
    return value


def _build_client() -> OpenAI:
    if OpenAI is Any:
        raise RuntimeError("The openai package is required for grader proxy calls.")
    return OpenAI(
        base_url=_required_env("API_BASE_URL"),
        api_key=_required_env("API_KEY"),
    )


def _build_environment(task_name: str):
    if LOCAL_IMAGE_NAME and RlFinanceEnv is not None:
        try:
            return RlFinanceEnv.from_docker_image(LOCAL_IMAGE_NAME)
        except Exception:
            pass
    return RlFinanceEnvironment(task_mode=task_name)


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
    env = _build_environment(task_name)
    rewards: list[float] = []
    banned_action_keys: set[str] = set()
    banned_targets: set[str] = set()
    seen_signatures: dict[tuple[str, float], tuple[str, int]] = {}
    steps = 0
    success = False
    _emit_start(task_name)
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
            _emit_step(steps, action_str, reward, done, error)
            success = done and reward > 0
    finally:
        close_method = getattr(env, "close", None)
        if callable(close_method):
            close_method()
        _emit_end(success, steps, rewards)
    return success


def run_inference(task_mode: str | None = None) -> list[tuple[str, bool]]:
    requested_mode = (task_mode or os.getenv("TASK_MODE", "easy")).strip().lower()
    modes: Iterable[str]
    if requested_mode == "all":
        modes = ("easy", "medium", "hard")
    elif requested_mode == "random":
        modes = (random.choice(("easy", "medium", "hard")),)
    elif requested_mode in SUPPORTED_TASK_MODES:
        modes = (requested_mode,)
    else:
        raise ValueError("task mode must be one of: easy, medium, hard, random, all")

    client = _build_client()
    return [(mode, run_episode(mode, client)) for mode in modes]


def main(argv: list[str] | None = None) -> int:
    parser = StructuredArgumentParser(add_help=False)
    parser.add_argument(
        "--task-mode",
        "--task",
        "--task-name",
        "--task_name",
        "--mode",
        default=None,
        dest="task_mode",
        help="Choose which task mode to run. Defaults to TASK_MODE env var or easy.",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="show_help",
        help="Show this help message and exit.",
    )
    cli_args = argv if argv is not None else os.sys.argv[1:]
    try:
        args, unknown_args = parser.parse_known_args(cli_args)
        if args.show_help:
            parser.print_help()
            return 0
        requested_task_mode = args.task_mode or _task_mode_from_unknown_args(unknown_args)
        run_inference(task_mode=requested_task_mode)
    except Exception as exc:
        requested_task_mode = None
        if "args" in locals():
            requested_task_mode = getattr(args, "task_mode", None)
        if not requested_task_mode and "unknown_args" in locals():
            requested_task_mode = _task_mode_from_unknown_args(unknown_args)
        return _emit_startup_failure(exc, requested_task_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 
 
