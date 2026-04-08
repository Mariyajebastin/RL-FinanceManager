# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the RL Finance environment.
"""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
except ImportError:  # pragma: no cover
    class OpenEnvAction(BaseModel):
        pass
    class OpenEnvObservation(BaseModel):
        done: bool = Field(default=False)
        reward: float | None = Field(default=None)
        metadata: dict[str, Any] = Field(default_factory=dict)

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

class TransactionView(BaseModel):
    """A masked view of a transaction. The agent sees this, not the true category."""
    transaction_id: str
    date: str
    amount: float
    description: str

class RlFinanceAction(OpenEnvAction):
    """STRICT Hackathon Action Space."""
    action_type: Literal["Categorize", "FlagDuplicate", "SuggestCut"] = Field(
        ..., description="The specific action the agent is taking."
    )
    transaction_id: str | None = Field(default=None)
    category: str | None = Field(default=None)
    percentage: float | None = Field(default=None)

class RlFinanceObservation(OpenEnvObservation):
    """STRICT Hackathon Observation Space."""
    current_balance: float = Field(..., description="The user's current bank balance.")
    recent_transactions: list[TransactionView] = Field(..., description="List of masked recent bank transactions.")
    current_task_objective: str = Field(..., description="The specific goal the agent must achieve.")