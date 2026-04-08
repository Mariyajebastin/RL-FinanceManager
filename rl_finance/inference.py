import os
import json
from openai import OpenAI
from pydantic import ValidationError

# Import the environment and models directly
from server.rl_finance_environment import PersonalFinanceEnv
from models import RlFinanceAction

# ==========================================
# 1. ENVIRONMENT VARIABLES & SETUP
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ==========================================
# 2. STRICT LOGGING UTILITIES
# ==========================================
def log_start(task_name: str, env_name: str, model_name: str):
    print(f"[START] task={task_name} env={env_name} model={model_name}")

def log_step(step_n: int, action_str: str, reward: float, done: bool, error: str = None):
    err_str = error if error else "null"
    done_str = "true" if done else "false"
    # Format reward to exactly 2 decimal places as required
    print(f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={done_str} error={err_str}")

def log_end(success: bool, steps: int, rewards: list):
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}")

# ==========================================
# 3. INFERENCE LOOP
# ==========================================
def run_inference():
    # Initialize the environment
    env = PersonalFinanceEnv(data_path="mock_data.json")
    obs = env.reset()
    
    task_name = "finance_manager"
    log_start(task_name=task_name, env_name="openenv-finance", model_name=MODEL_NAME)
    
    step_n = 0
    rewards = []
    done = False
    
    # Define the strict system prompt
    system_prompt = (
            "You are an expert AI Personal Finance Agent. Read the 'current_task_objective' carefully and ONLY take the action requested.\n"
            "- If the task is Easy (Categorize): Look at the single un-categorized or target transaction and use action_type 'Categorize'.\n"
            "- If the task is Medium (Duplicate): Look for identical subscription charges within a few days of each other and use 'FlagDuplicate'.\n"
            "- If the task is Hard (Suggest Cut): Find the category with the most discretionary spending (like Dining) and suggest cutting it. Use whole numbers for percentages (e.g., use 10.0 for 10%, NOT 0.1).\n"
            "Return ONLY the requested JSON format."
        )

    while not done:
        step_n += 1
        action_str = ""
        reward = 0.0
        error = None
        
        try:
            # Enforce structured output via OpenAI Beta parsing
            response = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Current State:\n{obs.model_dump_json()}"}
                ],
                response_format=RlFinanceAction,
            )
            
            # The AI's response is already perfectly mapped to our Pydantic class
            action = response.choices[0].message.parsed
            
            # Format the action string strictly for the stdout log
            if action.action_type == "Categorize":
                action_str = f"Categorize('{action.transaction_id}','{action.category}')"
            elif action.action_type == "FlagDuplicate":
                action_str = f"FlagDuplicate('{action.transaction_id}')"
            elif action.action_type == "SuggestCut":
                action_str = f"SuggestCut('{action.category}',{action.percentage})"
            else:
                action_str = f"{action.action_type}()"
            
            # Pass the Action object into the environment step
            obs, reward, done, info = env.step(action)
            
            if info.get("error"):
                error = info["error"]
                
        except ValidationError:
            # Handle AI hallucinating outside the Pydantic schema
            error = "Pydantic Validation Error - Bad LLM formatting"
            action_str = "InvalidAction"
            reward = -0.5
            done = True 
        except Exception as e:
            # Catch API outages or other fatal errors safely
            error = str(e).replace("\n", " ") 
            action_str = "SystemError"
            reward = 0.0
            done = True
            
        rewards.append(reward)
        log_step(step_n, action_str, reward, done, error)
        
    # Determine success (e.g., positive total reward indicates task completion)
    success = sum(rewards) > 0.0
    log_end(success, step_n, rewards)

if __name__ == "__main__":
    # Ensure standard output buffers are flushed immediately to prevent validation timing issues
    import sys
    import warnings
    warnings.filterwarnings("ignore") # Suppress external package warnings that might corrupt stdout
    
    try:
        run_inference()
    except Exception as fatal:
        # Guarantee [END] is printed even if the script catastrophically crashes
        print(f"[END] success=false steps=0 rewards=0.00")
        sys.exit(1)