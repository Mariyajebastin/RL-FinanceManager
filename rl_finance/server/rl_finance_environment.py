# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
from typing import Tuple, Dict, Any

# Import the strictly typed models we just finalized
from models import (
    RlFinanceObservation,
    RlFinanceAction,
    TransactionView,
    TransactionTruth
)

class PersonalFinanceEnv:
    def __init__(self, data_path: str = "mock_data.json"):
        """Initialize the environment and load the dataset."""
        self.data_path = data_path
        self.transactions_truth: Dict[str, TransactionTruth] = {}
        
        # State tracking
        self.current_step = 0
        self.max_steps = 15  # Prevent infinite loops
        self.done = False
        self.current_task_objective = ""
        self.current_balance = 5430.00 # Mock starting balance

    def reset(self) -> RlFinanceObservation:
        """Load a fresh profile and return the initial masked state."""
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
        
        # 4. Randomly assign a task for this episode
        tasks = [
            "Easy: Categorize the transaction.",
            "Medium: Identify and flag the duplicate subscription charge.",
            "Hard: Analyze spending and suggest a specific category to cut by 10%."
        ]
        self.current_task_objective = random.choice(tasks)
        
        return self._get_observation()

    def step(self, action: RlFinanceAction) -> Tuple[RlFinanceObservation, float, bool, Dict[str, Any]]:
        """Apply the action, calculate reward, and return the new state."""
        reward = 0.0
        info = {"error": None}
        self.current_step += 1

        if self.done:
            info["error"] = "Episode is already done. Please reset()."
            return self._get_observation(), 0.0, True, info

        try:
            # Route the action to the correct Grader
            if action.action_type == "Categorize":
                reward = self._grade_categorize(action)
            elif action.action_type == "FlagDuplicate":
                reward = self._grade_flag_duplicate(action)
            elif action.action_type == "SuggestCut":
                reward = self._grade_suggest_cut(action)
                self.done = True  # The Hard task acts as a final submission
            else:
                raise ValueError("Unknown action_type")

        except Exception as e:
            info["error"] = str(e)
            reward = -0.5  # Small penalty for formatting errors or invalid IDs

        # End the episode if the agent is looping too much
        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_observation(), float(reward), self.done, info

    # ==========================================
    # DETERMINISTIC TASK GRADERS
    # ==========================================

    def _grade_categorize(self, action: RlFinanceAction) -> float:
        """Grader for the Easy Task."""
        if not action.transaction_id or action.transaction_id not in self.transactions_truth:
            raise ValueError("Invalid transaction_id provided.")
            
        truth = self.transactions_truth[action.transaction_id]
        
        # CRUCIAL PENALTY: If they try to categorize the fraudulent duplicate instead of flagging it!
        # In our mock_data.json, TXN_044 is the 2nd Hulu charge.
        if action.transaction_id == "TXN_044":
            return -1.0
            
        # Standard grading: +0.1 for correct, 0.0 for incorrect
        if action.category and action.category.lower() == truth.true_category.lower():
            return 0.1
        return 0.0

    def _grade_flag_duplicate(self, action: RlFinanceAction) -> float:
        """Grader for the Medium Task."""
        if not action.transaction_id:
            raise ValueError("transaction_id is required to flag a duplicate.")
            
        # TXN_044 is our mathematical target for the duplicate
        if action.transaction_id == "TXN_044":
            self.done = True # Task complete!
            return 2.0
        return -0.5 # Penalty for flagging a normal transaction

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

    # ==========================================
    # HELPER METHODS
    # ==========================================

    def _get_observation(self) -> RlFinanceObservation:
        """Constructs the strictly masked observation for the AI."""
        masked_txns = []
        for txn_id, truth in self.transactions_truth.items():
            masked_txns.append(
                TransactionView(
                    transaction_id=truth.transaction_id,
                    date=truth.date,
                    amount=truth.amount,
                    description=truth.description
                )
            )
            
        return RlFinanceObservation(
            current_balance=self.current_balance,
            recent_transactions=masked_txns,
            current_task_objective=self.current_task_objective
        )