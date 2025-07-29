from abc import ABC, abstractmethod
from typing import Literal
import random

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from rating import RatingSystem, RatingPlayer


OptimizationMethod = Literal["linear", "hungarian"]


class MatchMaking(ABC):
    def __init__(
        self,
        rating_system: RatingSystem,
        optimization_method: OptimizationMethod,
        verbose: bool = False,
    ) -> None:
        self.rating_system = rating_system
        self.verbose = verbose
        self.optimization_method = optimization_method

        self.match_pairs = None

    @abstractmethod
    def weight_function(self, player: RatingPlayer, opponent: RatingPlayer) -> float:
        pass

    @abstractmethod
    def opponent_printing_function(
        self, player: RatingPlayer, opponent: RatingPlayer, weight: float
    ) -> str:
        pass

    def match_player_linearly(
        self,
        player: RatingPlayer,
        unmatched_players: dict[str, RatingPlayer],
        players: dict[str, RatingPlayer],
    ) -> str:

        if self.verbose:
            print(
                "- Looking for opponent to "
                f"{player['name']} with rating {player['rating']}"
            )

        player_name = player["name"]
        candidates: list[str] = []
        weights: list[float] = []

        for opponent_name in unmatched_players.keys():
            weight = self.weight_function(players[player_name], players[opponent_name])

            if self.verbose and self.opponent_printing_function is not None:
                text = self.opponent_printing_function(
                    players[player_name], players[opponent_name], weight
                )
                print(text)

            candidates.append(opponent_name)
            weights.append(weight)

        best_candidate_name = random.choices(candidates, weights=weights, k=1)[0]

        if self.verbose and self.opponent_printing_function is not None:
            text = self.opponent_printing_function(
                players[player_name], players[best_candidate_name], weight
            )

            print(f"- Selected: {text}\n")

        return best_candidate_name

    def match_players_linearly(self) -> None:

        players = {name: player for name, player in self.rating_system.players.items()}

        # Avoid a bias based on the position of the player
        keys = list(players.keys())
        random.shuffle(keys)
        shuffled_players = {key: players[key] for key in keys}

        unmatched_players = {
            player_name: player for player_name, player in shuffled_players.items()
        }
        matched_pairs: list[tuple[str, str]] = []

        while len(unmatched_players) > 1:
            player_name, player = unmatched_players.popitem()

            best_candidate_name = self.match_player_linearly(
                player, unmatched_players, players
            )

            _ = unmatched_players.pop(best_candidate_name)
            matched_pairs.append((player_name, best_candidate_name))

        # If still one there, compare it to all
        if unmatched_players:
            last_player_name, last_player = unmatched_players.popitem()
            players_without_last = {
                player_name: player
                for (player_name, player) in players.items()
                if player_name != last_player_name
            }

            best_candidate_name = self.match_player_linearly(
                last_player, players_without_last, players
            )

            matched_pairs.append((last_player_name, best_candidate_name))

        self.matched_pairs = matched_pairs

    def match_players_hungarian(self) -> None:
        # Using the Hungarian algorithm

        matched_pairs = []
        player_to_drop = None
        players = self.rating_system.players.copy()

        # If odd number match one randomly with linear approach
        if len(players) % 2 != 0:
            random_player = random.choice(list(players.keys()))
            players_without_random = {
                player_name: player
                for player_name, player in players.items()
                if player_name != random_player
            }
            matched_player = self.match_player_linearly(
                players[random_player], players_without_random, players
            )
            matched_pairs.append((random_player, matched_player))
            player_to_drop = random_player
            players.pop(player_to_drop)

        # Remaining players as a list
        players = list(players.values())

        num_players = len(players)
        cost_matrix = np.zeros((num_players, num_players))

        # Cost matrix, only upper triangle
        for i in range(num_players):
            for j in range(num_players):
                if i == j:
                    cost_matrix[i][j] = np.inf
                else:
                    weight = self.weight_function(players[i], players[j])
                    # Here, invert of cost
                    cost_matrix[i][j] = -weight

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = set()

        for i, j in zip(row_ind, col_ind):
            if i not in matched and j not in matched:
                matched.add(i)
                matched.add(j)
                matched_pairs.append((players[i]["name"], players[j]["name"]))

        self.matched_pairs = matched_pairs

    def match_players(self) -> None:

        if self.optimization_method == "linear":
            self.match_players_linearly()
        elif self.optimization_method == "hungarian":
            self.match_players_hungarian()
        else:
            raise KeyError(f"'{self.optimization_method}' not supported")

    def evaluate_match_making_rating_differences(self) -> tuple[pd.Series, pd.Series]:

        rating_differences: list[float] = []

        for player_1_name, player_2_name in self.matched_pairs:
            rating_differences.append(
                self.rating_system.players[player_1_name]["rating"]
                - self.rating_system.players[player_2_name]["rating"]
            )

        rating_diff_series = pd.Series(rating_differences)
        rating_abs_diff_series = rating_diff_series.abs()

        if self.verbose:
            print("Pairs Rating Difference:")
            print(rating_diff_series.describe())
            print("Absolute Pairs Rating Difference:")
            print(rating_abs_diff_series.describe())

            plt.figure()
            plt.hist(rating_diff_series, bins=20)
            plt.title(f"Pairs Ratings Differences Distribution")
            plt.show()

        return rating_diff_series, rating_abs_diff_series

    def evaluate_match_making_winning_probability_differences(self) -> pd.DataFrame:

        winning_probabilities_differences: list[float] = []

        for player_1_name, player_2_name in self.matched_pairs:
            win_p, loose_p, draw_p = self.rating_system.compute_winning_odds(
                player_1_name, player_2_name
            )
            winning_probabilities_differences.append(win_p - loose_p)

        rating_diff_series = pd.Series(winning_probabilities_differences)
        rating_abs_diff_series = rating_diff_series.abs()

        if self.verbose:
            print("Pairs Winning Probability Difference:")
            print(rating_diff_series.describe())
            print("Absolute Pairs Winning Probability Difference:")
            print(rating_abs_diff_series.describe())

            plt.figure()
            plt.hist(rating_diff_series, bins=20)
            plt.title(f"Pairs Winning Probability Differences Distribution")
            plt.show()

        return rating_diff_series, rating_abs_diff_series
