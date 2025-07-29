# Note: This implementation is based on the following paper:
# https://inria.hal.science/inria-00149859/document

import math

from rating import RatingPlayer, Games, RatingSystem


# Recommended values for the paper
STARTING_RATING = 1.0

# Values to define convergence and boundaries
DEFAULT_CONVERGENCE_TOLERANCE = 1e-3
MAXIMUM_ITERATIONS = 1e6


def compute_new_rating_iteration(
    player_rating: float, player_wins: int, opponent_ratings: list[float]
) -> float:
    # Computing one iteration of one player rating

    numerator = player_wins

    denominator = 0.0
    for opponent_rating in opponent_ratings:
        denominator += player_rating / (player_rating + opponent_rating)

    # In case no game played, keep same rating
    if denominator == 0.0:
        return player_rating

    new_rating = numerator / denominator

    return new_rating


def compute_new_ratings_iteration(
    games: Games,
    previous_players_ratings: dict[str, float],
    players_nb_wins: dict[str, int],
) -> dict[str, float]:
    # Compute one iteration of all players ratings

    # Do not overwrite with new results as first all the ratings have to be updated
    new_players_ratings: dict[str, float] = {}

    for player_name, player_rating in previous_players_ratings.items():
        opponents_rating: list[float] = []
        for game in games:
            if game["player_a_name"] == player_name:
                opponents_rating.append(previous_players_ratings[game["player_b_name"]])
            elif game["player_b_name"] == player_name:
                opponents_rating.append(previous_players_ratings[game["player_a_name"]])

        new_players_ratings[player_name] = compute_new_rating_iteration(
            player_rating, players_nb_wins[player_name], opponents_rating
        )

    return new_players_ratings


def compute_nb_wins_per_player(games: Games, player_names: list[str]) -> dict[str, int]:

    players_nb_wins: dict[str, int] = {player_name: 0 for player_name in player_names}

    for game in games:
        players_nb_wins[game["player_a_name"]] += game["outcome"]
        players_nb_wins[game["player_b_name"]] += 1 - game["outcome"]

    return players_nb_wins


def compute_ratings_evolution(
    previous_players_ratings: dict[str, float], players_ratings: dict[str, float]
) -> float:

    ratings_change = 0.0

    for player_name, previous_rating in previous_players_ratings.items():
        ratings_change += math.pow(previous_rating - players_ratings[player_name], 2)

    ratings_change = math.sqrt(ratings_change)

    return ratings_change


class SimpleBayesianPlayer(RatingPlayer):
    pass


class SimpleBayesianSystem(RatingSystem):

    def __init__(
        self,
        starting_rating: float = STARTING_RATING,
        convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
        verbose: bool = False,
    ) -> None:
        super().__init__(starting_rating=starting_rating, verbose=verbose)
        self.players: dict[str, SimpleBayesianPlayer]

        self.convergence_tolerance = convergence_tolerance

    def create_new_player(self, name: str) -> SimpleBayesianPlayer:
        return SimpleBayesianPlayer(name=name, rating=self.starting_rating)

    def create_player(self, name: str, rating: float) -> SimpleBayesianPlayer:
        return SimpleBayesianPlayer(name=name, rating=rating)

    def update_ratings(self, games: Games) -> None:
        # Reminder: For SimpleBayesian, the order of the games does NOT matter!

        self.check_players_in_games(games)

        previous_players_ratings = {
            player_name: player["rating"]
            for player_name, player in self.players.items()
        }
        players_nb_wins = compute_nb_wins_per_player(games, list(self.players.keys()))

        nb_iterations = 0
        ratings_evolution = math.inf

        # Adjust the tolerance to take into account that more players, harder to reach
        adjusted_tolerance = self.convergence_tolerance * len(previous_players_ratings)

        while ratings_evolution >= adjusted_tolerance:
            players_ratings = compute_new_ratings_iteration(
                games, previous_players_ratings, players_nb_wins
            )
            ratings_evolution = compute_ratings_evolution(
                previous_players_ratings, players_ratings
            )

            previous_players_ratings = players_ratings

            nb_iterations += 1
            if nb_iterations > MAXIMUM_ITERATIONS:
                raise ValueError("Couldn't converge.")

        if self.verbose:
            print(f"Converged in {nb_iterations} iterations.")

        for player_name in self.players:
            self.players[player_name]["rating"] = previous_players_ratings[player_name]

    def compute_winning_odds(
        self, player_a_name: str, player_b_name: str, **kwargs
    ) -> tuple[float, float, float]:
        # Always 0.0 for draw.

        player_a = self.players[player_a_name]
        player_b = self.players[player_b_name]

        winning_probability = player_a["rating"] / (
            player_a["rating"] + player_b["rating"]
        )
        loosing_probability = 1 - winning_probability
        drawing_probability = 1 - winning_probability - loosing_probability

        if self.verbose:
            print(f"Winning probability: {100*winning_probability:.2f}")
            print(f"Loosing probability: {100*loosing_probability:.2f}")
            print(f"Drawing probability: {100*drawing_probability:.2f}")

        return winning_probability, loosing_probability, drawing_probability
