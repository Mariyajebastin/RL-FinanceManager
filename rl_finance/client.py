# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rl Finance Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import RlFinanceAction, RlFinanceObservation


class RlFinanceEnv(
    EnvClient[RlFinanceAction, RlFinanceObservation, State]
):
    """
    Client for the Rl Finance Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with RlFinanceEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(RlFinanceAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = RlFinanceEnv.from_docker_image("rl_finance-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(RlFinanceAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: RlFinanceAction) -> Dict:
        """
        Convert RlFinanceAction to JSON payload for step message.

        Args:
            action: RlFinanceAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RlFinanceObservation]:
        """
        Parse server response into StepResult[RlFinanceObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with RlFinanceObservation
        """
        obs_data = payload.get("observation", {})
        observation = RlFinanceObservation(
            current_balance=obs_data.get("current_balance", 0.0),
            recent_transactions=obs_data.get("recent_transactions", []),
            current_task_objective=obs_data.get("current_task_objective", ""),
            last_action_failed=obs_data.get("last_action_failed", False),
            current_page=obs_data.get("current_page", 0),
            total_pages=obs_data.get("total_pages", 1),
            total_transactions=obs_data.get("total_transactions", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
