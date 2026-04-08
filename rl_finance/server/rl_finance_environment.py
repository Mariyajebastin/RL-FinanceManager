# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import random
import uuid
from typing import Any, Dict, Tuple

try:
    from ..models import (
        RlFinanceObservation,
        RlFinanceAction,
        RlFinanceState,
        TransactionView,
        TransactionTruth,
    )
except ImportError:  # pragma: no cover
    from models import (
        RlFinanceObservation,
        RlFinanceAction,
        RlFinanceState,
        TransactionView,
        TransactionTruth,
    )

class RlFinanceEnvironment:
    def __init__(self, data_name: str = "mock_data.json", task_mode: str | None = None):
        """Initialize the environment and load the dataset."""
        # This finds the file in the ROOT regardless of where the script is run
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = os.path.join(base_path, data_name)
        self.transactions_truth: Dict[str, TransactionTruth] = {}
        
        # State tracking
        self.current_step = 0
        self.max_steps = 30  # Prevent infinite loops but allow pagination
        self.done = False
        self.current_task_objective = ""
        self.current_balance = 5430.00 # Mock starting balance
        self.last_action_failed = False
        self.last_error: str | None = None
        self.episode_id: str | None = None
        
        # Pagination
        self.current_page = 0
        self.page_size = 10
        configured_task_mode = task_mode if task_mode is not None else os.getenv("TASK_MODE", "random")
        self.task_mode = configured_task_mode.strip().lower()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **_: Any,
    ) -> RlFinanceObservation:
        """Load a fresh profile and return the initial masked state."""
        if seed is not None:
            random.seed(seed)
        # 1. Load the pristine JSON
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
            
        # 2. Store the TRUTH internally so we can grade the agent
        self.transactions_truth = {
            t["transaction_id"]: TransactionTruth(**t) for t in raw_data
        }
        
        # 3. Reset state variables
        self.current_step = 0
        self.done = False
        self.current_page = 0
        self.page_size = 10
        self.last_action_failed = False
        self.last_error = None
        self.episode_id = episode_id or str(uuid.uuid4())
        
        # 4. Randomly assign a task for this episode
        tasks = {
            "easy": "Easy: Categorize the transaction.",
            "medium": "Medium: Identify and flag the duplicate subscription charge.",
            "hard": "Hard: Analyze spending and suggest a specific category to cut by 10%.",
        }
        self.current_task_objective = tasks.get(self.task_mode, random.choice(list(tasks.values())))
        
        return self._get_observation()

    def step(
        self,
        action: RlFinanceAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> RlFinanceObservation:
        """Apply the action, calculate reward, and return the new state."""
        reward = 0.0
        info = {"error": None}
        self.current_step += 1

        if self.done:
            info["error"] = "Episode is already done. Please reset()."
            self.last_error = info["error"]
            return self._build_step_observation(0.0, True)
            
        # --- NEW PAGINATION LOGIC ---
        if action.action_type == "NextPage":
            total_transactions = len(self.transactions_truth)
            max_page_index = max(math.ceil(total_transactions / self.page_size) - 1, 0)
            
            if self.current_page < max_page_index:
                self.current_page += 1
                reward = -0.05  # Tiny penalty: Discourages infinite scrolling, but cheaper than a wrong guess
            else:
                reward = -0.50  # Big penalty: Tried to scroll past the end of the document
                
            self.last_action_failed = reward < 0
            self.done = False if self.current_step < self.max_steps else True
            self.last_error = info["error"]
            return self._build_step_observation(reward, self.done)
        # ----------------------------

        # --- CATEGORIZE INLINE HANDLER (with informative errors) ---
        if action.action_type == "Categorize":
            if not action.category or not action.transaction_id:
                info["error"] = "Categorize failed: You MUST provide both 'transaction_id' AND 'category' fields."
                reward = -0.10
                self.done = False if self.current_step < self.max_steps else True
                self.last_error = info["error"]
                return self._build_step_observation(reward, self.done)
            
            if action.transaction_id not in self.transactions_truth:
                info["error"] = f"Unknown transaction_id: {action.transaction_id}"
                reward = -0.10
                self.done = False if self.current_step < self.max_steps else True
                self.last_error = info["error"]
                return self._build_step_observation(reward, self.done)
            
            if action.transaction_id == "TXN_044":
                info["error"] = "TXN_044 is a duplicate, not a category. Use FlagDuplicate instead."
                reward = -0.50
            else:
                truth = self.transactions_truth[action.transaction_id]
                if action.category.lower() == truth.true_category.lower():
                    truth.is_categorized = True
                    reward = 0.10
                    info["error"] = None
                else:
                    info["error"] = f"Incorrect. '{action.category}' is the wrong category for {action.transaction_id}."
                    reward = -0.05
            
            self.done = True if (reward > 0 or self.current_step >= self.max_steps) else False
            self.last_action_failed = True if reward < 0 else False
            self.last_error = info["error"]
            return self._build_step_observation(float(reward), self.done)
        # -----------------------------------------------------------

        # --- SUGGESTCUT INLINE HANDLER (with informative errors) ---
        if action.action_type == "SuggestCut":
            if not action.category or action.percentage is None:
                info["error"] = "SuggestCut requires both 'category' (string) and 'percentage' (float)."
                reward = -0.10
                self.done = False if self.current_step < self.max_steps else True
                self.last_error = info["error"]
                return self._build_step_observation(reward, self.done)

            reward = self._grade_suggest_cut(action)

            if reward > 0:
                self.done = True
                info["error"] = None
            else:
                info["error"] = (
                    f"Incorrect. '{action.category}' at {action.percentage:.2f}% does not match the expected"
                    " savings recommendation."
                )
                reward = -0.05
                self.done = False if self.current_step < self.max_steps else True
            
            self.last_action_failed = True if reward < 0 else False
            self.last_error = info["error"]
            return self._build_step_observation(float(reward), self.done)
        # -----------------------------------------------------------

        try:
            # Route the action to the correct Grader
            if action.action_type == "FlagDuplicate":
                # _grade_flag_duplicate returns (reward, error_msg)
                reward, flag_error = self._grade_flag_duplicate(action)
                if flag_error:
                    info["error"] = flag_error
            else:
                raise ValueError("Unknown action_type")

        except Exception as e:
            info["error"] = str(e)
            reward = -0.5  # Small penalty for formatting errors or invalid IDs

        # STAY ALIVE RULE: End the episode ONLY if success (reward > 0) or max steps reached
        if reward > 0 or self.current_step >= self.max_steps:
            self.done = True
        else:
            self.done = False

        self.last_action_failed = True if reward < 0 else False
        self.last_error = info["error"]
        return self._build_step_observation(float(reward), self.done)

    # ==========================================
    # DETERMINISTIC TASK GRADERS
    # ==========================================

    def _grade_categorize(self, action: RlFinanceAction) -> float:
        """Grader for the Easy Task."""
        if not action.category or not action.transaction_id or action.transaction_id not in self.transactions_truth:
            return -0.50 # Penalize missing data
            
        truth = self.transactions_truth[action.transaction_id]
        
        # CRUCIAL PENALTY: If they try to categorize the fraudulent duplicate instead of flagging it!
        # In our mock_data.json, TXN_044 is the 2nd Hulu charge.
        if action.transaction_id == "TXN_044":
            return -1.0
            
        # Standard grading: +0.10 for correct, -0.10 for incorrect
        if action.category and action.category.lower() == truth.true_category.lower():
            # Mark it as categorized in your state so it disappears from the observation!
            truth.is_categorized = True
            return 0.10 # Positive signal for partial progress!
        else:
            return -0.10 # Wrong category

    def _grade_flag_duplicate(self, action: RlFinanceAction) -> tuple:
        """Grader for the Medium Task. Returns (reward, error_msg)."""
        if not action.transaction_id:
            return (-0.05, "transaction_id is required to flag a duplicate.")
            
        # TXN_044 is our mathematical target for the duplicate
        if action.transaction_id == "TXN_044":
            self.done = True # Task complete!
            return (1.0, None)
        return (-0.05, f"Incorrect. {action.transaction_id} is not the duplicate subscription charge. Keep looking.")

    def _grade_suggest_cut(self, action: RlFinanceAction) -> float:
        """Grader for the Hard Task."""
        if not action.category or action.percentage is None:
            raise ValueError("SuggestCut requires both category and percentage.")
            
        # The prompt requires a strict mathematical check. 
        # The target was 10% of discretionary spending (Dining).
        is_correct_category = action.category.lower() == "dining"
        is_correct_math = abs(action.percentage - 10.0) < 0.01  # Safe float comparison
        
        if is_correct_category and is_correct_math:
            return 1.0
        return 0.0

    @property
    def state(self) -> RlFinanceState:
        """Return the current environment state for OpenEnv-compatible inspection."""
        return RlFinanceState(
            episode_id=self.episode_id,
            step_count=self.current_step,
            task_mode=self.task_mode,
            current_task_objective=self.current_task_objective,
            current_page=self.current_page,
            max_steps=self.max_steps,
        )

    async def reset_async(self) -> RlFinanceObservation:
        """Async wrapper for UI/runtime compatibility."""
        return self.reset()

    async def step_async(
        self, action: RlFinanceAction
    ) -> Tuple[RlFinanceObservation, float, bool, Dict[str, Any]]:
        """Async wrapper for UI/runtime compatibility."""
        return self.step(action)

    async def state_async(self) -> RlFinanceState:
        """Async wrapper for UI/runtime compatibility."""
        return self.state

    def close(self) -> None:
        """No-op close hook for OpenEnv server compatibility."""
        return None

    # ==========================================
    # HELPER METHODS
    # ==========================================

    # In rl_finance_environment.py
    def _build_step_observation(self, reward: float, done: bool) -> RlFinanceObservation:
        observation = self._get_observation()
        observation.reward = reward
        observation.done = done
        observation.metadata = {"error": self.last_error}
        return observation

    def _get_observation(self) -> RlFinanceObservation:
        # Get ALL transactions
        all_txns = list(self.transactions_truth.values())
        
        # Only show transactions that haven't been categorized yet
        pending_txns = [t for t in all_txns if not getattr(t, 'is_categorized', False)]
        
        # Calculate the start and end indices for the current page
        start_idx = self.current_page * self.page_size
        end_idx = start_idx + self.page_size
        
        # Slice the data!
        current_batch = pending_txns[start_idx:end_idx]
        total_pending = len(pending_txns)
        total_pages = max(math.ceil(total_pending / self.page_size), 1)
        
        views = [TransactionView(
            transaction_id=t.transaction_id, 
            date=t.date, 
            amount=t.amount, 
            description=t.description
        ) for t in current_batch]
        
        return RlFinanceObservation(
            current_balance=self.current_balance,
            recent_transactions=views,
            current_task_objective=self.current_task_objective,
            last_action_failed=self.last_action_failed,
            current_page=self.current_page,
            total_pages=total_pages,
            total_transactions=total_pending,
            metadata={"error": self.last_error},
        )


PersonalFinanceEnv = RlFinanceEnvironment
