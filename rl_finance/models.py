# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the RL Finance environment.

This module keeps the starter OpenEnv action/observation models while adding
the finance domain entities described by the ER diagram.
"""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
except ImportError:  # pragma: no cover
    class OpenEnvAction(BaseModel):
        """Fallback action model used when openenv is not installed."""

    class OpenEnvObservation(BaseModel):
        """Fallback observation model used when openenv is not installed."""
        done: bool = Field(default=False)
        reward: float | None = Field(default=None)
        metadata: dict[str, Any] = Field(default_factory=dict)


# ==========================================
# INTERNAL STATE MODELS (Hidden from Agent)
# ==========================================

class User(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    name: str = Field(..., description="Display name for the user")

class Category(BaseModel):
    name: str = Field(..., description="Category name such as Dining or Groceries")
    description: str | None = Field(default=None)

class Anomaly(BaseModel):
    type: str = Field(..., description="Anomaly type")
    reason: str = Field(..., description="Why the transaction was flagged")

class Transaction(BaseModel):
    """INTERNAL transaction containing the ground-truth answer."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    description: str = Field(..., description="Merchant or transaction description")
    amount: float = Field(..., description="Signed transaction amount")
    date: str = Field(..., description="Transaction date")
    type: Literal["credit", "debit"] = Field(..., description="Transaction direction")
    true_category: str = Field(..., description="Ground-truth category label (HIDDEN FROM AGENT)")
    category: Category | None = Field(default=None)
    anomalies: list[Anomaly] = Field(default_factory=list)

class EpisodeAction(BaseModel):
    """Internal log of actions generated within an episode."""
    predicted_category: str = Field(..., description="Predicted category label")
    flag_type: str | None = Field(default=None)
    suggestion: str | None = Field(default=None)
    confidence: float = Field(..., ge=0.0, le=1.0)

class EpisodeState(BaseModel):
    """Internal state snapshot recorded during an episode."""
    step: int = Field(..., ge=0)
    reward: float = Field(..., description="Reward assigned at the step")
    flagged_items: list[str] = Field(default_factory=list)

class Episode(BaseModel):
    """An RL episode backend tracking transactions, actions, and state history."""
    episode_id: str = Field(..., description="Unique episode identifier")
    user_id: str = Field(..., description="Owner of the episode")
    task_type: str = Field(..., description="Task performed in the episode")
    step_count: int = Field(default=0, ge=0)
    total_reward: float = Field(default=0.0)
    transactions: list[Transaction] = Field(default_factory=list)
    actions: list[EpisodeAction] = Field(default_factory=list)
    states: list[EpisodeState] = Field(default_factory=list)

    def add_transaction(self, transaction: Transaction) -> None:
        self.transactions.append(transaction)

    def add_action(self, action: EpisodeAction) -> None:
        self.actions.append(action)

    def add_state(self, state: EpisodeState) -> None:
        self.states.append(state)
        self.sync_metrics()

    def sync_metrics(self) -> None:
        self.step_count = len(self.states)
        self.total_reward = sum(state.reward for state in self.states)


# ==========================================
# AGENT I/O MODELS (OpenEnv Specs)
# ==========================================

class TransactionView(BaseModel):
    """A masked view of a transaction. The agent sees this, not the true category."""
    transaction_id: str
    date: str
    amount: float
    description: str

class RlFinanceAction(OpenEnvAction):
    """
    STRICT Hackathon Action Space.
    """
    action_type: Literal["Categorize", "FlagDuplicate", "SuggestCut"] = Field(
        ..., description="The specific action the agent is taking."
    )
    transaction_id: str | None = Field(
        default=None, description="Required for Categorize and FlagDuplicate."
    )
    category: str | None = Field(
        default=None, description="Required for Categorize and SuggestCut."
    )
    percentage: float | None = Field(
        default=None, description="Required for SuggestCut. (e.g., 10.0 for 10%)."
    )

class RlFinanceObservation(OpenEnvObservation):
    """
    STRICT Hackathon Observation Space.
    Notice we DO NOT include the full Episode or Transaction here to prevent cheating.
    """
    current_balance: float = Field(..., description="The user's current bank balance.")
    recent_transactions: list[TransactionView] = Field(
        ..., description="List of masked recent bank transactions."
    )
    current_task_objective: str = Field(
        ..., description="The specific goal the agent must achieve."
    )