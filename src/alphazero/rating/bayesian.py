# Note: This implementation is based on the following blog:
# https://www.remi-coulom.fr/Bayesian-Elo/
# The blog didn't provide a lot of details, and thus results might differ
# Mainly, a standard optimized is used, not the minorization-maximization algorithm

import math

import scipy
import numpy as np
import scipy.optimize

from rating import RatingPlayer, Games, RatingSystem


STARTING_RATING = 1000.0
DEFAULT_LOW_RATING_BOUNDARY = 500.0
DEFAULT_HIGH_RATING_BOUNDARY = 2500.0

# Recommended values for the blog (specific to chess!)
DEFAULT_FIRST_ADVANTAGE = 32.8
DEFAULT_DRAW_BIAS = 97.3


def compute_raw_expected_outcome(difference: float) -> float:

    expected_outcome_inverse = 1 + math.pow(10, difference / 400)
    expected_outcome = 1 / expected_outcome_inverse

    return expected_outcome


def compute_player_1_winning_probability(
    player_1_rating: float,
    player_2_rating: float,
    first_advantage: float = DEFAULT_FIRST_ADVANTAGE,
    draw_bias: float = DEFAULT_DRAW_BIAS,
) -> float:
    return compute_raw_expected_outcome(
        -player_1_rating + player_2_rating - first_advantage + draw_bias
    )


def compute_player_2_winning_probability(
    player_1_rating: float,
    player_2_rating: float,
    first_advantage: float = DEFAULT_FIRST_ADVANTAGE,
    draw_bias: float = DEFAULT_DRAW_BIAS,
) -> float:
    return compute_raw_expected_outcome(
        player_1_rating - player_2_rating + first_advantage + draw_bias
    )


def compute_drawing_probability(
    player_1_winning_probability: float, player_2_winning_probability: float
) -> float:
    return 1 - player_1_winning_probability - player_2_winning_probability


def compute_ratings_log_likelihood(
    ratings: list[float],
    first_player_wins: np.ndarray,
    second_player_wins: np.ndarray,
    players_draws: np.ndarray,
    first_advantage: float = DEFAULT_FIRST_ADVANTAGE,
    draw_bias: float = DEFAULT_DRAW_BIAS,
    verbose: bool = False,
) -> float:
    # Compute the likelihood of the given players ratings

    total_log_likelihood = 0.0

    for first_player_win in first_player_wins:
        player_1_id, player_2_id = first_player_win

        player_1_win_probability = compute_player_1_winning_probability(
            ratings[player_1_id], ratings[player_2_id], first_advantage, draw_bias
        )

        total_log_likelihood += math.log(player_1_win_probability)

    for second_player_win in second_player_wins:
        player_1_id, player_2_id = second_player_win

        player_2_win_probability = compute_player_2_winning_probability(
            ratings[player_1_id], ratings[player_2_id], first_advantage, draw_bias
        )

        total_log_likelihood += math.log(player_2_win_probability)

    for player_draw in players_draws:
        player_1_id, player_2_id = player_draw

        win_probability = compute_player_1_winning_probability(
            ratings[player_1_id], ratings[player_2_id], first_advantage, draw_bias
        )
        loose_probability = compute_player_2_winning_probability(
            ratings[player_1_id], ratings[player_2_id], first_advantage, draw_bias
        )
        draw_probability = compute_drawing_probability(
            win_probability, loose_probability
        )

        total_log_likelihood += 0.5 * math.log(draw_probability)

    # if verbose:
    #     print(f"Current log-likelihood: {total_log_likelihood}")

    return total_log_likelihood


def optimize_ratings(
    ratings: list[float],
    first_player_wins: np.ndarray,
    second_player_wins: np.ndarray,
    players_draws: np.ndarray,
    first_advantage: float = DEFAULT_FIRST_ADVANTAGE,
    draw_bias: float = DEFAULT_DRAW_BIAS,
    low_elo_boundary: float = DEFAULT_LOW_RATING_BOUNDARY,
    high_elo_boundary: float = DEFAULT_HIGH_RATING_BOUNDARY,
    verbose: bool = False,
) -> list[float]:

    def objective_function(ratings):
        return -compute_ratings_log_likelihood(
            ratings,
            first_player_wins,
            second_player_wins,
            players_draws,
            first_advantage,
            draw_bias,
            verbose,
        )

    boundaries = []
    for _ in ratings:
        boundaries.append((low_elo_boundary, high_elo_boundary))

    optimized_ratings = scipy.optimize.minimize(
        objective_function, ratings, bounds=boundaries
    )

    if verbose:
        print("Number of iterations:", optimized_ratings.nit)

    optimized_ratings = optimized_ratings.x
    optimized_ratings = optimized_ratings.tolist()

    return optimized_ratings


def create_wins_and_draws_tables(
    games: Games, players_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    player_name_to_index = {
        player_name: index for index, player_name in enumerate(players_names)
    }

    first_player_wins = []
    second_player_wins = []
    players_draws = []

    for game in games:
        first_player = game["player_a_name"]
        second_player = game["player_b_name"]

        if game["outcome"] == 1.0:
            first_player_wins.append(
                [
                    player_name_to_index[first_player],
                    player_name_to_index[second_player],
                ]
            )
        if game["outcome"] == 0.0:
            second_player_wins.append(
                [
                    player_name_to_index[first_player],
                    player_name_to_index[second_player],
                ]
            )
        if game["outcome"] == 0.5:
            players_draws.append(
                [
                    player_name_to_index[first_player],
                    player_name_to_index[second_player],
                ]
            )

    first_player_wins = np.array(first_player_wins)
    second_player_wins = np.array(second_player_wins)
    players_draws = np.array(players_draws)

    return first_player_wins, second_player_wins, players_draws


class BayesianPlayer(RatingPlayer):
    pass


class BayesianSystem(RatingSystem):

    def __init__(
        self,
        starting_rating: float = STARTING_RATING,
        first_advantage: float = DEFAULT_FIRST_ADVANTAGE,
        draw_bias: float = DEFAULT_DRAW_BIAS,
        low_elo_boundary=DEFAULT_LOW_RATING_BOUNDARY,
        high_elo_boundary=DEFAULT_HIGH_RATING_BOUNDARY,
        verbose: bool = False,
    ) -> None:
        super().__init__(starting_rating=starting_rating, verbose=verbose)
        self.players: dict[str, BayesianPlayer]

        self.first_advantage = first_advantage
        self.draw_bias = draw_bias
        self.low_elo_boundary = low_elo_boundary
        self.high_elo_boundary = high_elo_boundary

    def create_new_player(self, name: str) -> BayesianPlayer:
        return BayesianPlayer(name=name, rating=self.starting_rating)

    def create_player(self, name: str, rating: float) -> BayesianPlayer:
        return BayesianPlayer(name=name, rating=rating)

    def update_ratings(self, games: Games) -> None:
        # Reminder: For Bayesian, the order of the games does NOT matter!
        # Players that didn't play in this session will keep their ratings unchanged

        self.check_players_in_games(games)

        players_names = list(self.players.keys())
        first_player_wins, second_player_wins, players_draws = (
            create_wins_and_draws_tables(games, players_names)
        )

        players_ratings = [player["rating"] for player in self.players.values()]
        optimized_ratings = optimize_ratings(
            players_ratings,
            first_player_wins,
            second_player_wins,
            players_draws,
            self.first_advantage,
            self.draw_bias,
            self.low_elo_boundary,
            self.high_elo_boundary,
            self.verbose,
        )

        for index, player_name in enumerate(self.players.keys()):
            self.players[player_name]["rating"] = optimized_ratings[index]

    def compute_winning_odds(
        self, player_a_name: str, player_b_name: str, is_first: bool
    ) -> tuple[float, float, float]:

        player_a = self.players[player_a_name]
        player_b = self.players[player_b_name]

        if is_first:
            winning_probability = compute_player_1_winning_probability(
                player_a["rating"],
                player_b["rating"],
                self.first_advantage,
                self.draw_bias,
            )
            loosing_probability = compute_player_2_winning_probability(
                player_a["rating"],
                player_b["rating"],
                self.first_advantage,
                self.draw_bias,
            )
            drawing_probability = compute_drawing_probability(
                winning_probability, loosing_probability
            )
        else:
            winning_probability = compute_player_2_winning_probability(
                player_b["rating"],
                player_a["rating"],
                self.first_advantage,
                self.draw_bias,
            )
            loosing_probability = compute_player_1_winning_probability(
                player_b["rating"],
                player_a["rating"],
                self.first_advantage,
                self.draw_bias,
            )
            drawing_probability = compute_drawing_probability(
                winning_probability, loosing_probability
            )

        if self.verbose:
            print(f"Winning probability: {100*winning_probability:.2f}")
            print(f"Loosing probability: {100*loosing_probability:.2f}")
            print(f"Drawing probability: {100*drawing_probability:.2f}")

        return winning_probability, loosing_probability, drawing_probability
