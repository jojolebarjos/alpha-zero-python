from abc import ABC, abstractmethod
from typing import TypedDict

import pandas as pd
import matplotlib.pyplot as plt


class RatingPlayer(TypedDict):
    name: str
    rating: float


# A game between two players
# The outcome is be 1 if player A wins, 0 if player loses, 0.5 if draw
# If order matters, player_a is the first one to play
class Game(TypedDict):
    player_a_name: str
    player_b_name: str
    outcome: float


Games = list[Game]


class RatingSystem(ABC):
    def __init__(self, starting_rating: float, verbose: bool = False) -> None:
        # Starting rating is the rating a new player starts with

        self.starting_rating = starting_rating
        self.verbose = verbose

        self.players: dict[str, RatingPlayer] = {}

    def create_new_player(self, name: str) -> RatingPlayer:
        return RatingPlayer(name=name, rating=self.starting_rating)

    def create_player(self, name: str, rating: float) -> RatingPlayer:
        return RatingPlayer(name=name, rating=rating)

    def add_player(self, name: str, rating: float | None = None, **kwargs) -> None:
        if name in self.players:
            raise KeyError(f"{name} already used by another player.")

        if rating is None:
            self.players[name] = self.create_new_player(name=name)
        else:
            self.players[name] = self.create_player(name=name, rating=rating, **kwargs)

    def remove_player(self, player_name: str) -> None:
        self.players.pop(player_name)

    def display_players(self) -> None:
        print("Ratings:")
        for player_name, player in self.players.items():
            print(f"{player_name}: {player['rating']:.2f}")
        print()

    def check_players_in_games(self, games: Games) -> None:
        for game in games:
            if game["player_a_name"] not in self.players:
                raise ValueError(
                    f"Player '{game['player_a_name']}' not found in players list"
                )
            if game["player_b_name"] not in self.players:
                raise ValueError(
                    f"Player '{game['player_b_name']}' not found in players list"
                )

    @abstractmethod
    def update_ratings(self, games: Games, **kwargs) -> None:
        self.check_players_in_games(games)
        pass

    @abstractmethod
    def compute_winning_odds(
        self, player_a_name: str, player_b_name: str, **kwargs
    ) -> tuple[float, float, float]:
        # Return [proba of A winning, proba of A loosing, proba of draw]
        pass


def compare_rating_systems_players_rating(
    ratings_systems: list[RatingSystem],
    player_names: list[str],
    corrective_addition_factors: list[float] | None = None,
    corrective_multiplication_factors: list[float] | None = None,
    plot: bool = True,
) -> pd.DataFrame:

    if corrective_multiplication_factors is None:
        corrective_multiplication_factors = [1.0 for _ in ratings_systems]
    if corrective_addition_factors is None:
        corrective_addition_factors = [0.0 for _ in ratings_systems]

    players_ratings_in_rating_system: dict[str, dict[str, float]] = {}

    for i, rating_system in enumerate(ratings_systems):
        rating_system_name = rating_system.__class__.__name__
        if rating_system_name not in players_ratings_in_rating_system:
            players_ratings_in_rating_system[rating_system_name] = {}
        for player_name in player_names:
            players_ratings_in_rating_system[rating_system_name][player_name] = (
                rating_system.players[player_name]["rating"]
                * corrective_multiplication_factors[i]
                + corrective_addition_factors[i]
            )

    df_ratings = pd.DataFrame.from_dict(
        players_ratings_in_rating_system, orient="index"
    )

    if plot:
        plt.figure()
        for rating_system in ratings_systems:
            name = rating_system.__class__.__name__
            ratings = df_ratings.loc[name]
            plt.hist(ratings, bins=20, label=name, alpha=0.5)
        plt.title("Ratings Distribution per Method")
        plt.legend()
        plt.show()

    return df_ratings


def compare_rating_systems_players_rating_pair(
    df_ratings: pd.DataFrame,
    rating_system_a_name: str,
    rating_system_b_name: str,
    corrective_addition_factors: list[float] | None = None,
    corrective_multiplication_factors: list[float] | None = None,
) -> None:

    if corrective_multiplication_factors is None:
        corrective_multiplication_factors = [1.0 for _ in range(2)]
    if corrective_addition_factors is None:
        corrective_addition_factors = [0.0 for _ in range(2)]

    ratings_a = (
        df_ratings.loc[rating_system_a_name] * corrective_multiplication_factors[0]
        + corrective_addition_factors[0]
    )
    ratings_b = (
        df_ratings.loc[rating_system_b_name] * corrective_multiplication_factors[1]
        + corrective_addition_factors[1]
    )

    ratings_difference = ratings_a - ratings_b
    abs_ratings_difference = ratings_difference.abs()

    print("Rating Difference:")
    print(ratings_difference.describe())
    print("Absolute Rating Difference:")
    print(abs_ratings_difference.describe())

    plt.figure()
    plt.hist(ratings_a, bins=20, label=rating_system_a_name, alpha=0.5)
    plt.hist(ratings_b, bins=20, label=rating_system_b_name, alpha=0.5)
    plt.title(f"Ratings Distribution: {rating_system_a_name} VS {rating_system_b_name}")
    plt.legend()
    plt.show()


def compare_rating_systems_players_rank(
    ratings_systems: list[RatingSystem], player_names: list[str]
) -> pd.DataFrame:

    players_rank_in_rating_system: dict[str, dict[str, int]] = {}

    for rating_system in ratings_systems:
        rating_system_name = rating_system.__class__.__name__
        if rating_system_name not in players_rank_in_rating_system:
            players_rank_in_rating_system[rating_system_name] = {}

        name_rating_pairs = {
            name: player["rating"] for name, player in rating_system.players.items()
        }
        sorted_names = sorted(
            name_rating_pairs, key=name_rating_pairs.get, reverse=True
        )
        name_to_rank = {name: rank + 1 for rank, name in enumerate(sorted_names)}

        for player_name in player_names:
            players_rank_in_rating_system[rating_system_name][player_name] = (
                name_to_rank[player_name]
            )

    df_ranking = pd.DataFrame.from_dict(players_rank_in_rating_system, orient="index")

    return df_ranking


def compare_rating_systems_players_ranking_pair(
    df_ranking: pd.DataFrame, rating_system_a_name: str, rating_system_b_name: str
) -> None:

    rankings_a = df_ranking.loc[rating_system_a_name]
    rankings_b = df_ranking.loc[rating_system_b_name]

    rankings_difference = rankings_a - rankings_b
    abs_rankings_difference = rankings_difference.abs()

    print("Ranking Difference:")
    print(rankings_difference.describe())
    print("Absolute Ranking Difference:")
    print(abs_rankings_difference.describe())

    plt.figure()
    plt.hist(rankings_difference, bins=20)
    plt.title(
        f"Ranking Differences Distribution: {rating_system_a_name} VS {rating_system_b_name}"
    )
    plt.show()


def compare_rating_systems_winning_odds(
    ratings_systems: list[RatingSystem],
    player_a_name: str,
    player_b_name: str,
    **kwargs,
) -> pd.DataFrame:

    winning_probas: list[float] = []
    loosing_probas: list[float] = []
    drawing_probas: list[float] = []

    for rating_system in ratings_systems:
        winning_proba, loosing_proba, drawing_proba = (
            rating_system.compute_winning_odds(player_a_name, player_b_name, **kwargs)
        )

        winning_probas.append(winning_proba)
        loosing_probas.append(loosing_proba)
        drawing_probas.append(drawing_proba)

    df_winning_odds = pd.DataFrame(
        {
            "winning_proba": winning_probas,
            "loosing_proba": loosing_probas,
            "drawing_proba": drawing_probas,
        },
        index=[
            f"{rating_system.__class__.__name__}" for rating_system in ratings_systems
        ],
    )
    df_winning_odds *= 100

    return df_winning_odds
