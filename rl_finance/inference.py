import os, sys, json, warnings, re, argparse

# Force the current directory into the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from pydantic import ValidationError

try:
    from .server.rl_finance_environment import RlFinanceEnvironment
    from .models import RlFinanceAction
except ImportError:
    from server.rl_finance_environment import RlFinanceEnvironment
    from models import RlFinanceAction

warnings.filterwarnings("ignore")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
if "Llama-3-8B-Instruct:fastest" in MODEL_NAME: MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN: raise ValueError("HF_TOKEN required")
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def log_step(step_n, action_str, reward, done, error=None):
    print(f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={'true' if done else 'false'} error={error or 'null'}")
    sys.stdout.flush()

def run_episode(task_mode: str | None = None):
    env = RlFinanceEnvironment(task_mode=task_mode)
    obs = env.reset()
    print(
        f"[START] task=finance_manager env=openenv-finance model={MODEL_NAME} mode={task_mode or env.task_mode} objective={obs.current_task_objective}",
        flush=True,
    )
    
    step_n, rewards, done = 0, [], False
    # Strict JSON formatting for smaller models
    system_prompt = (
        "You are a strict Financial AI. Output ONLY ONE valid JSON object.\n"
        f"OBJECTIVE: {obs.current_task_objective}\n\n"
        "CRITICAL JSON RULES:\n"
        "1. 'action_type' MUST be exactly: 'Categorize', 'FlagDuplicate', 'SuggestCut', or 'NextPage'.\n"
        "2. For 'Categorize': you MUST include 'transaction_id' AND 'category'. No exceptions.\n"
        "3. For 'FlagDuplicate': you MUST include 'transaction_id' with a proper ID like 'TXN_044'.\n"
        "4. Use 'NextPage' when the evidence you need is not visible on the current page.\n"
        "5. For duplicate detection, compare merchants/descriptions and identical amounts before guessing.\n"
        "6. DO NOT use code comments (// or /*). Put all thoughts in 'reasoning'.\n"
        "7. Output exactly ONE JSON object. Never output two.\n\n"
        "EXAMPLE CATEGORIZE (copy this format!):\n"
        '{"reasoning": "TXN_001 is a salary deposit.", "action_type": "Categorize", "transaction_id": "TXN_001", "category": "Income"}'
    )
    
    # Initialize a small scratchpad and blacklist
    history = ""
    failed_ids = []
    
    last_reward = 0.0

    while not done:
        step_n += 1
        try:
            # Convert the transactions into a clean, readable text list
            txn_list_text = "\n".join([f"- ID: {t.transaction_id} | {t.description} | {t.amount}" for t in obs.recent_transactions])
            
            # Inject Blacklist Warning
            blacklist_warning = ""
            if failed_ids:
                blacklist_warning = f"CRITICAL: Previously failed IDs (DO NOT USE): {failed_ids}\n"

            # Combine EVERYTHING into a readable text block with Post-Prompt Rules
            user_msg = (
                f"{history}\n"
                f"{blacklist_warning}\n"
                f"Objective: {obs.current_task_objective}\n\n"
                f"Page: {obs.current_page + 1} of {obs.total_pages}\n"
                f"Visible transactions on this page: {len(obs.recent_transactions)} of {obs.total_transactions}\n\n"
                f"Transactions to scan:\n{txn_list_text}\n\n"
                "--- CRITICAL REMINDER ---\n"
                "1. Output ONLY valid JSON starting with '{' and ending with '}'.\n"
                "2. 'transaction_id' MUST be a proper ID like 'TXN_044'. Do NOT use descriptions.\n"
                "3. Match your action to the stated objective. Do not always choose FlagDuplicate.\n"
                "4. If the duplicate is not visible yet, use NextPage instead of guessing random IDs."
            )
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ]
            )
            
            content = response.choices[0].message.content
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                clean_json = match.group(0)
                action = RlFinanceAction.model_validate_json(clean_json)
            else:
                raise ValueError("No JSON object found in AI response")
                
            if action.action_type == "SuggestCut":
                action_str = f"SuggestCut('{action.category or ''}', {action.percentage if action.percentage is not None else 'null'})"
            elif action.action_type == "Categorize":
                action_str = f"Categorize('{action.transaction_id or ''}', '{action.category or ''}')"
            elif action.action_type == "FlagDuplicate":
                action_str = f"FlagDuplicate('{action.transaction_id or ''}')"
            else:
                action_str = action.action_type
            obs, reward, done, info = env.step(action)
            error = info.get("error")
            last_reward = reward
            
            # Update Blacklist on failure
            if reward < 0 and action.transaction_id:
                if action.transaction_id not in failed_ids:
                    failed_ids.append(action.transaction_id)

            if reward > 0:
                done = True
                
            if error:
                history = f"PREVIOUS ERROR: Step {step_n} failed with error: {error}."
            else:
                history = f"Step {step_n} gave Reward {reward:.2f}"
                
        except ValidationError as e:
            # STOP THE BLEEDING: End immediately if hallucinating
            action_str, reward, done, error = "InvalidAction", -1.0, True, f"Format Fail: {e.json()}"
            last_reward = reward
        except Exception as e:
            action_str, reward, done, error = "Error", 0.0, True, str(e).replace("\n", " ")
            last_reward = reward
            
        rewards.append(reward)
        log_step(step_n, action_str, reward, done, error)
    
    success = "true" if done and last_reward > 0 else "false"
    score = max(rewards) if rewards else 0.0
    print(f"[END] success={success} steps={step_n} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}", flush=True)
    return success == "true"


def run_inference(task_mode: str | None = None):
    requested_mode = (task_mode if task_mode is not None else os.getenv("TASK_MODE", "random")).strip().lower()

    if requested_mode == "all":
        results = []
        for mode in ("easy", "medium", "hard"):
            results.append((mode, run_episode(mode)))
        summary = " ".join([f"{mode}={'pass' if ok else 'fail'}" for mode, ok in results])
        print(f"[SUMMARY] {summary}", flush=True)
        return

    run_episode(requested_mode if requested_mode else None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-mode",
        choices=["random", "easy", "medium", "hard", "all"],
        default=None,
        help="Choose which task mode to run. Defaults to TASK_MODE env var or random.",
    )
    args = parser.parse_args()

    try:
        run_inference(task_mode=args.task_mode)
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=0.00", flush=True)
 
 