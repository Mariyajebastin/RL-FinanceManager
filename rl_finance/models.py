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


class User(BaseModel):
    """A user who runs RL finance episodes."""

    user_id: str = Field(..., description="Unique identifier for the user")
    name: str = Field(..., description="Display name for the user")


class Category(BaseModel):
    """A normalized finance category linked to transactions."""

    name: str = Field(..., description="Category name such as Dining or Groceries")
    description: str | None = Field(
        default=None,
        description="Optional description for the category",
    )


class Anomaly(BaseModel):
    """An anomaly detected for a transaction."""

    type: str = Field(..., description="Anomaly type")
    reason: str = Field(..., description="Why the transaction was flagged")


class Transaction(BaseModel):
    """A transaction processed during an episode."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    description: str = Field(..., description="Merchant or transaction description")
    amount: float = Field(..., description="Signed transaction amount")
    date: str = Field(..., description="Transaction date")
    type: Literal["credit", "debit"] = Field(
        ...,
        description="Transaction direction",
    )
    true_category: str = Field(..., description="Ground-truth category label")
    category: Category | None = Field(
        default=None,
        description="Resolved category entity for the transaction",
    )
    anomalies: list[Anomaly] = Field(
        default_factory=list,
        description="Optional anomalies associated with the transaction",
    )


class EpisodeAction(BaseModel):
    """An action generated within an episode."""

    predicted_category: str = Field(..., description="Predicted category label")
    flag_type: str | None = Field(
        default=None,
        description="Type of issue or anomaly flag raised by the model",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested follow-up or correction",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the action",
    )


class EpisodeState(BaseModel):
    """A state snapshot recorded during an episode."""

    step: int = Field(..., ge=0, description="Current step number")
    reward: float = Field(..., description="Reward assigned at the step")
    flagged_items: list[str] = Field(
        default_factory=list,
        description="Transaction IDs or labels flagged at this step",
    )


class Episode(BaseModel):
    """An RL episode containing transactions, actions, and state history."""

    episode_id: str = Field(..., description="Unique episode identifier")
    user_id: str = Field(..., description="Owner of the episode")
    task_type: str = Field(..., description="Task performed in the episode")
    step_count: int = Field(default=0, ge=0, description="Number of processed steps")
    total_reward: float = Field(
        default=0.0,
        description="Sum of rewards accumulated across states",
    )
    transactions: list[Transaction] = Field(
        default_factory=list,
        description="Transactions assigned to the episode",
    )
    actions: list[EpisodeAction] = Field(
        default_factory=list,
        description="Actions generated during the episode",
    )
    states: list[EpisodeState] = Field(
        default_factory=list,
        description="State snapshots maintained through the episode",
    )

    def add_transaction(self, transaction: Transaction) -> None:
        """Attach a transaction to the episode."""

        self.transactions.append(transaction)

    def add_action(self, action: EpisodeAction) -> None:
        """Record a generated action for the episode."""

        self.actions.append(action)

    def add_state(self, state: EpisodeState) -> None:
        """Record a state snapshot and refresh episode metrics."""

        self.states.append(state)
        self.sync_metrics()

    def sync_metrics(self) -> None:
        """Keep episode totals aligned with the recorded state history."""

        self.step_count = len(self.states)
        self.total_reward = sum(state.reward for state in self.states)


class RlFinanceAction(OpenEnvAction):
    """
    OpenEnv action payload.

    `message` is kept for backward compatibility with the starter echo
    environment, while the finance fields support the ER-diagram workflow.
    """

    message: str = Field(default="", description="Legacy message payload")
    transaction_id: str | None = Field(
        default=None,
        description="Transaction being acted on",
    )
    predicted_category: str | None = Field(
        default=None,
        description="Predicted category for the current transaction",
    )
    flag_type: str | None = Field(
        default=None,
        description="Optional anomaly or review flag",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested follow-up action",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for the prediction",
    )


class RlFinanceObservation(OpenEnvObservation):
    """
    OpenEnv observation payload.

    Legacy echo fields are preserved so the current starter environment keeps
    working while the richer finance graph can be populated by future server
    logic.
    """

    echoed_message: str = Field(default="", description="Legacy echoed message")
    message_length: int = Field(default=0, description="Length of echoed message")
    user: User | None = Field(
        default=None,
        description="User associated with the current episode",
    )
    episode: Episode | None = Field(
        default=None,
        description="Current finance episode",
    )
    current_transaction: Transaction | None = Field(
        default=None,
        description="Transaction currently being processed",
    )
    latest_action: EpisodeAction | None = Field(
        default=None,
        description="Most recent action generated during the episode",
    )
    latest_state: EpisodeState | None = Field(
        default=None,
        description="Most recent episode state snapshot",
    )
    categories: list[Category] = Field(
        default_factory=list,
        description="Available categories for transaction labeling",
    )
    anomalies: list[Anomaly] = Field(
        default_factory=list,
        description="Detected anomalies relevant to the observation",
    )
