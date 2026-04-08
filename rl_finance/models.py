# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the RL Finance environment.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
    from openenv.core.env_server.types import State as OpenEnvState
except ImportError:  # pragma: no cover
    class OpenEnvAction(BaseModel):
        pass
    class OpenEnvObservation(BaseModel):
        done: bool = Field(default=False)
        reward: float | None = Field(default=None)
        metadata: dict[str, Any] = Field(default_factory=dict)
    class OpenEnvState(BaseModel):
        episode_id: str | None = Field(default=None)
        step_count: int = Field(default=0)

class User(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    name: str = Field(..., description="Display name for the user")

class Category(BaseModel):
    name: str = Field(..., description="Category name such as Dining or Groceries")
    description: str | None = Field(default=None)

class Anomaly(BaseModel):
    type: str = Field(..., description="Anomaly type")
    reason: str = Field(..., description="Why the transaction was flagged")

class TransactionTruth(BaseModel):
    """INTERNAL transaction containing the ground-truth answer."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    description: str = Field(..., description="Merchant or transaction description")
    amount: float = Field(..., description="Signed transaction amount")
    date: str = Field(..., description="Transaction date")
    type: Literal["credit", "debit"] = Field(..., description="Transaction direction")
    true_category: str = Field(..., description="Ground-truth category label (HIDDEN FROM AGENT)")
    category: Category | None = Field(default=None)
    anomalies: list[Anomaly] = Field(default_factory=list)
    is_categorized: bool = Field(default=False, description="Tracks if agent has correctly categorized this transaction")

class TransactionView(BaseModel):
    """A masked view of a transaction. The agent sees this, not the true category."""
    transaction_id: str
    date: str
    amount: float
    description: str

class RlFinanceAction(OpenEnvAction):
    """STRICT Hackathon Action Space."""
    # This allows the model to ignore extra junk fields the AI sends
    model_config = {"extra": "ignore"} 

    # Add a default for reasoning so the validation doesn't crash if the AI skips it
    reasoning: str = Field(default="No reasoning provided", description="Step-by-step logic.")

    # Keep this required, but ensure the prompt emphasizes it
    action_type: Literal["Categorize", "FlagDuplicate", "SuggestCut", "NextPage"] = Field(
        ..., 
        description="MUST be exactly 'Categorize', 'FlagDuplicate', 'SuggestCut', or 'NextPage'. Use NextPage to scroll data."
    )

    transaction_id: Optional[str] = Field(default=None, description="The ID of the transaction (e.g., 'TXN_044').")
    category: Optional[str] = Field(default=None)
    percentage: Optional[float] = Field(default=None)

class RlFinanceObservation(OpenEnvObservation):
    """STRICT Hackathon Observation Space."""
    current_balance: float = Field(..., description="The user's current bank balance.")
    recent_transactions: list[TransactionView] = Field(..., description="List of masked recent bank transactions.")
    current_task_objective: str = Field(..., description="The specific goal the agent must achieve.")
    last_action_failed: bool = Field(default=False, description="Flag indicating if the last action was incorrect.")
    current_page: int = Field(default=0, description="Zero-based page index for the current transaction window.")
    total_pages: int = Field(default=1, description="Total number of pages available for the current transaction window.")
    total_transactions: int = Field(default=0, description="Total number of masked transactions available in the episode.")


class RlFinanceState(OpenEnvState):
    task_mode: str = Field(default="random", description="Current task mode for the episode.")
    current_task_objective: str = Field(default="", description="Objective assigned to the current episode.")
    current_page: int = Field(default=0, description="Current page index.")
    max_steps: int = Field(default=30, description="Maximum steps allowed in the episode.")
