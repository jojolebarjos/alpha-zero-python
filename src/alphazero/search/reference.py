from __future__ import annotations

import math

import numpy as np

from alphazero.data import Action, State, Prediction


class Search:
    """Graph search.

    This is the key algorithm behind AlphaZero. It is a modified version of MCTS, which
    includes neural network predictions. The search is guided by a predicted prior
    policy, and the rollout is replaced by a predicted value.

    To allow batched inference, the algorithm is split in half. First, the selection
    of a state is done. Then, once the model output is acquired, the prediction is
    fed to complete the search iteration.

    """

    def __init__(self, initial_state: State, c_puct: float) -> None:
        self.initial_state = initial_state
        self.c_puct = c_puct
        self.state_nodes: dict[State, StateNode] = {}
        self.selected_state_node: StateNode | None = None

    def select(self) -> bool:
        """Select state.

        This step goes down the tree, and selects a (usually unseen) state to visit. In
        some cases, when a terminal node is reached or a cycle is found, the descent
        does not pick any state to evaluate. This still counts as an iteration.

        """

        # If a selection is pending, this is an error
        if self.selected_state_node is not None:
            raise RuntimeError

        # Start at root node
        state_node = self.state_nodes.get(self.initial_state)

        # Edge case, initial state must be evaluated at first step
        if state_node is None:
            assert not self.initial_state.has_ended
            state_node = StateNode(self.initial_state)
            self.state_nodes[self.initial_state] = state_node
            self.selected_state_node = state_node
            return True

        # Self reference, to avoid cycle through root node
        assert state_node.last_state_node is None
        state_node.last_state_node = state_node

        while True:
            # If this is a terminal node, we just backtrack with the true reward
            if state_node.state.has_ended:
                assert state_node.raw_value is not None
                self._backpropagate(state_node)
                return False

            # Select best action using PUCT (i.e., deterministic)
            action_node = state_node.select_action_node(self.c_puct)
            action_node.visit_count += 1

            # If the action was never taken, compute the next game state
            next_state_node = action_node.next_state_node
            if next_state_node is None:
                next_state = action_node.action.sample_next_state()
                next_state_node = self.state_nodes.get(next_state)

                # If this is an unseen state, select it
                if next_state_node is None:
                    next_state_node = StateNode(next_state)
                    next_state_node.last_state_node = state_node
                    self.state_nodes[next_state] = next_state_node
                    action_node.next_state_node = next_state_node
                    if next_state.has_ended:
                        next_state_node.average_value = next_state_node.raw_value = next_state.reward
                        self._backpropagate(next_state_node)
                        return False
                    self.selected_state_node = next_state_node
                    return True

                # Otherwise, the state node did exist as a transposition; link and continue descent
                action_node.next_state_node = next_state_node

                # TODO if child visit is greater than edge visit, then maybe we can skip?
                # https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md#continuing-vs-stopping-playouts-when-child-visits--edge-visits

            # Check for cycle; if there is one, we abort
            if next_state_node.last_state_node is not None:
                self._backpropagate(state_node)
                return False

            # Recurse
            next_state_node.last_state_node = state_node
            state_node = next_state_node

    def feed(self, prediction: Prediction) -> None:
        """Feed prediction output.

        If an evaluation was requested, this method must be called with the model
        prediction. This saves the policy and value heads outputs, and backpropagate
        visit counts and average values.

        """

        # If no selection is pending, this is an error
        if self.selected_state_node is None:
            raise RuntimeError

        # Save prediction
        self.selected_state_node.action_nodes = [
            ActionNode(action, probability) for action, probability in zip(prediction.actions, prediction.policy)
        ]
        self.selected_state_node.raw_value = prediction.value

        # Propagate result, and reset path
        self._backpropagate(self.selected_state_node)
        self.selected_state_node = None

    def _backpropagate(self, state_node: StateNode) -> None:
        while True:
            if state_node.state.has_ended:
                state_node.visit_count += 1

            else:
                visit_count = 1
                average_value = state_node.raw_value.copy()
                for action_node in state_node.action_nodes:
                    visit_count += action_node.visit_count
                    if action_node.next_state_node is not None:
                        average_value += action_node.next_state_node.average_value * action_node.visit_count
                state_node.visit_count = visit_count
                state_node.average_value = average_value / state_node.visit_count

            if state_node.last_state_node is None:
                break

            if state_node.last_state_node is state_node:
                state_node.last_state_node = None
                break

            state_node.last_state_node, state_node = None, state_node.last_state_node

    @property
    def state(self) -> State:
        return self.selected_state_node.state

    @property
    def prediction(self) -> Prediction:
        """Posterior distribution."""

        state_node = self.state_nodes[self.initial_state]
        total_visit_count = sum(action_node.visit_count for action_node in state_node.action_nodes)
        actions = [action_node.action for action_node in state_node.action_nodes]
        policy = np.array([action_node.visit_count / total_visit_count for action_node in state_node.action_nodes])
        value = state_node.average_value
        return Prediction(actions, policy, value)

    # TODO allow rebasing tree to a specific node, to reuse part of the statistics


class StateNode:
    def __init__(self, state: State) -> None:
        self.state = state
        self.action_nodes: list[ActionNode] | None = None
        self.raw_value: np.ndarray | None = None
        self.average_value: np.ndarray | None = None
        self.visit_count: int = 0
        self.last_state_node: StateNode | None = None

    def select_action_node(self, c_puct: float) -> ActionNode:
        player = self.state.player

        # TODO is it always state_node.visit_count - 1?
        total_visit_count = sum(action_node.visit_count for action_node in self.action_nodes)

        best_score = -math.inf
        best_action_node = None
        for action_node in self.action_nodes:
            average_utility = 0.0
            if action_node.next_state_node is not None:
                average_utility = action_node.next_state_node.average_value[player]
            visit_scale = math.sqrt(total_visit_count) / (1 + action_node.visit_count)
            score = average_utility + c_puct * action_node.raw_policy * visit_scale
            if score > best_score:
                best_score = score
                best_action_node = action_node

        assert best_action_node is not None
        return best_action_node


class ActionNode:
    def __init__(self, action: Action, raw_policy: float) -> None:
        self.action = action
        self.raw_policy = raw_policy
        self.next_state_node: StateNode | None = None
        self.visit_count: int = 0
