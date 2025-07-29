# Note: the following is based on this paper: https://glicko.net/glicko/glicko2.pdf

import math

from rating import RatingPlayer, Games, RatingSystem


# Recommended values from the paper
STARTING_RATING = 0.0
STARTING_VOLATILITY = 0.06
VOLATILITY_TOLERANCE = 0.000001
DEFAULT_TAO = 0.5

# Constants to go from Glicko2 to Elo scale
FIDE_STARTING_RATING = 1000.0
DEFAULT_ELO_UNCERTAINTY = 350.0
GLICKO2_TO_ELO_SCALE = 400 / math.log(10)
STARTING_GLICKO2_UNCERTAINTY = DEFAULT_ELO_UNCERTAINTY / GLICKO2_TO_ELO_SCALE

# To stop in case of non-convergence
MAXIMUM_ITERATIONS_FOR_VOLATILITY = 1e6


def compute_rating_confidence(rating_uncertainty: float) -> float:

    return 1 / math.sqrt(
        1 + 3 * (math.pow(rating_uncertainty, 2) / (math.pow(math.pi, 2)))
    )


def compute_expected_outcome(
    player_rating: float, opponent_rating: float, opponent_confidence: float
) -> float:

    weighted_rating_diff = opponent_confidence * (player_rating - opponent_rating)
    expected_outcome = 1 / (1 + math.exp(-weighted_rating_diff))

    return expected_outcome


def compute_game_reliability(
    player_rating: float, opponent_rating: float, opponent_confidence: float
) -> float:

    expected_outcome = compute_expected_outcome(
        player_rating, opponent_rating, opponent_confidence
    )
    game_reliability = (
        math.pow(opponent_confidence, 2) * expected_outcome * (1 - expected_outcome)
    )

    return game_reliability


def compute_estimated_variance(
    player_rating: float,
    opponent_ratings: list[float],
    opponent_confidences: list[float],
) -> float:

    total_reliability = 0.0

    for opponent_rating, opponent_confidence in zip(
        opponent_ratings, opponent_confidences
    ):
        total_reliability += compute_game_reliability(
            player_rating, opponent_rating, opponent_confidence
        )

    estimated_variance = 1 / total_reliability

    return estimated_variance


def compute_game_estimated_improvement(
    player_rating: float,
    opponent_rating: float,
    opponent_confidence: float,
    outcome: float,
) -> float:

    expected_outcome = compute_expected_outcome(
        player_rating, opponent_rating, opponent_confidence
    )
    estimated_improvement = opponent_confidence * (outcome - expected_outcome)

    return estimated_improvement


def compute_estimated_improvement(
    player_rating: float,
    player_estimated_variance: float,
    opponent_ratings: list[float],
    opponent_confidences: list[float],
    outcomes: list[float],
) -> float:

    total_improvement = 0.0

    for opponent_rating, opponent_confidence, outcome in zip(
        opponent_ratings, opponent_confidences, outcomes
    ):
        improvement = compute_game_estimated_improvement(
            player_rating, opponent_rating, opponent_confidence, outcome
        )
        total_improvement += improvement

    estimated_improvement = player_estimated_variance * total_improvement

    return estimated_improvement


def f(
    x: float,
    estimated_improvement: float,
    rating_uncertainty: float,
    variance: float,
    a: float,
    tao: float = DEFAULT_TAO,
):
    # Function used to maximize the likelihood of the volatility, as in the paper

    first_numerator = math.exp(x) * (
        math.pow(estimated_improvement, 2)
        - math.pow(rating_uncertainty, 2)
        - variance
        - math.exp(x)
    )
    first_denominator = 2 * math.pow(
        (math.pow(rating_uncertainty, 2) + variance + math.exp(x)), 2
    )

    second_numerator = x - a
    second_denominator = math.pow(tao, 2)

    return first_numerator / first_denominator - second_numerator / second_denominator


def compute_volatility(
    estimated_improvement: float,
    rating_uncertainty: float,
    variance: float,
    previous_volatility: float,
    tao: float,
    verbose: bool = False,
) -> float:
    # Iterative function to find the new volatility, as in the paper

    safety_cnt = 0

    a = math.log(math.pow(previous_volatility, 2))
    A = a

    if math.pow(estimated_improvement, 2) > math.pow(rating_uncertainty, 2) + variance:
        B = math.log(
            math.pow(estimated_improvement, 2)
            - math.pow(rating_uncertainty, 2)
            - variance
        )
    else:
        k = 1
        x = a - k * tao
        while f(x, estimated_improvement, rating_uncertainty, variance, a, tao) < 0:
            k += 1
            x = a - k * tao

            safety_cnt += 1
            if safety_cnt > MAXIMUM_ITERATIONS_FOR_VOLATILITY:
                raise ValueError("Safety counter exceeded, check your inputs")

        B = a - k * tao

    safety_cnt = 0
    if verbose:
        print(f"Initial brackets for volatility: [{A}, {B}]")

    f_A = f(A, estimated_improvement, rating_uncertainty, variance, a, tao)
    f_B = f(B, estimated_improvement, rating_uncertainty, variance, a, tao)

    C = A + (A - B) * f_A / (f_B - f_A)
    f_C = f(C, estimated_improvement, rating_uncertainty, variance, a, tao)

    while abs(B - A) > VOLATILITY_TOLERANCE:
        if f_C * f_B <= 0:
            A = B
            f_A = f_B
        else:
            f_A = f_A / 2

        B = C
        f_B = f_C
        C = A + (A - B) * f_A / (f_B - f_A)
        f_C = f(C, estimated_improvement, rating_uncertainty, variance, a, tao)

        if verbose:
            print(f"Brackets for volatility: [{A}, {B}]")

        safety_cnt += 1
        if safety_cnt > MAXIMUM_ITERATIONS_FOR_VOLATILITY:
            raise ValueError("Safety counter exceeded, check your inputs")

    volatility = math.exp(A / 2)

    if verbose:
        print(f"Volatility: {volatility}\n")

    return volatility


def compute_new_rating_uncertainty(
    previous_rating_uncertainty: float, estimated_variance: float, volatility: float
) -> float:

    temporary_rating_uncertainty = math.sqrt(
        math.pow(previous_rating_uncertainty, 2) + math.pow(volatility, 2)
    )

    inverse_rating_uncertainty = math.sqrt(
        (1 / math.pow(temporary_rating_uncertainty, 2)) + (1 / estimated_variance)
    )

    rating_uncertainty = 1 / inverse_rating_uncertainty

    return rating_uncertainty


def compute_new_rating(
    previous_rating: float,
    rating_uncertainty: float,
    opponent_ratings: list[float],
    opponent_confidences: list[float],
    outcomes: list[float],
) -> float:

    total_estimated_improvement = 0.0

    for opponent_rating, opponent_confidence, outcome in zip(
        opponent_ratings, opponent_confidences, outcomes
    ):
        estimated_improvement = compute_game_estimated_improvement(
            previous_rating, opponent_rating, opponent_confidence, outcome
        )
        total_estimated_improvement += estimated_improvement

    rating = (
        previous_rating + math.pow(rating_uncertainty, 2) * total_estimated_improvement
    )

    return rating


def compute_new_metrics_based_on_games(
    previous_rating: float,
    previous_rating_uncertainty: float,
    previous_volatility: float,
    opponent_ratings: list[float],
    opponent_uncertainties: list[float],
    outcomes: list[float],
    tao: float = DEFAULT_TAO,
    verbose: bool = False,
) -> tuple[float, float, float]:

    opponent_confidences = [
        compute_rating_confidence(uncertainty) for uncertainty in opponent_uncertainties
    ]

    # If never played a game return only an increased uncertainty
    if len(outcomes) == 0:
        rating_uncertainty = math.sqrt(
            math.pow(previous_rating_uncertainty, 2) + math.pow(previous_volatility, 2)
        )
        return previous_rating, rating_uncertainty, previous_volatility

    variance = compute_estimated_variance(
        previous_rating, opponent_ratings, opponent_confidences
    )

    estimated_improvement = compute_estimated_improvement(
        previous_rating, variance, opponent_ratings, opponent_confidences, outcomes
    )

    volatility = compute_volatility(
        estimated_improvement,
        previous_rating_uncertainty,
        variance,
        previous_volatility,
        tao,
        verbose=verbose,
    )

    rating_uncertainty = compute_new_rating_uncertainty(
        previous_rating_uncertainty, variance, volatility
    )

    new_rating = compute_new_rating(
        previous_rating,
        rating_uncertainty,
        opponent_ratings,
        opponent_confidences,
        outcomes,
    )

    return new_rating, rating_uncertainty, volatility


def glicko2_to_elo(
    rating: float,
    rating_uncertainty: float,
    elo_starting_rating: float = FIDE_STARTING_RATING,
) -> tuple[float, float]:

    elo_rating = elo_starting_rating + GLICKO2_TO_ELO_SCALE * rating
    elo_uncertainty = GLICKO2_TO_ELO_SCALE * rating_uncertainty

    return elo_rating, elo_uncertainty


class Glicko2Player(RatingPlayer):
    rating_uncertainty: float
    volatility: float


class Glicko2System(RatingSystem):

    def __init__(
        self,
        starting_rating: float = STARTING_RATING,
        starting_uncertainty: float = STARTING_GLICKO2_UNCERTAINTY,
        starting_volatility: float = STARTING_VOLATILITY,
        tao: float = DEFAULT_TAO,
        verbose: bool = False,
    ) -> None:
        super().__init__(starting_rating=starting_rating, verbose=verbose)
        self.players: dict[str, Glicko2Player]

        self.tao = tao
        self.starting_uncertainty = starting_uncertainty
        self.starting_volatility = starting_volatility

    def create_new_player(self, name: str) -> Glicko2Player:
        return Glicko2Player(
            name=name,
            rating=self.starting_rating,
            rating_uncertainty=self.starting_uncertainty,
            volatility=self.starting_volatility,
        )

    def create_player(
        self, name: str, rating: float, rating_uncertainty: float, volatility: float
    ) -> Glicko2Player:
        return Glicko2Player(
            name=name,
            rating=rating,
            rating_uncertainty=rating_uncertainty,
            volatility=volatility,
        )

    def display_players(
        self, elo_starting_rating: float = FIDE_STARTING_RATING
    ) -> None:
        super().display_players()
        print("Uncertainty:")
        for player_name, player in self.players.items():
            print(f"{player_name}: {player['rating_uncertainty']:.2f}")
        print()
        print("Volatility:")
        for player_name, player in self.players.items():
            print(f"{player_name}: {player['volatility']:.8f}")
        print()

        # Also show in Elo scale
        ratings = []
        rating_uncertainties = []
        for player_name, player in self.players.items():
            elo_rating, elo_uncertainty = glicko2_to_elo(
                player["rating"], player["rating_uncertainty"], elo_starting_rating
            )
            ratings.append(elo_rating)
            rating_uncertainties.append(elo_uncertainty)

        print("Rating (Elo Scale):")
        for i, (player_name, player) in enumerate(self.players.items()):
            print(f"{player_name}: {ratings[i]}")
        print()
        print("Rating Uncertainty (Elo Scale):")
        for i, (player_name, player) in enumerate(self.players.items()):
            print(f"{player_name}: {rating_uncertainties[i]}")
        print()

    def update_ratings(self, games: Games) -> None:
        # Reminder: For Glicko2, the order of the games does NOT matter!
        # Players that didn't play in this session will have their rating
        # uncertainty changed

        self.check_players_in_games(games)

        # The players do not have to be updated directly, they are only updated once
        # all the batch has been processed
        updated_players: dict[str, Glicko2Player] = {}

        for player_name, player in self.players.items():
            opponent_ratings = []
            opponent_uncertainties = []
            outcomes = []

            # This could be improved if done previous by creating a dictionary of
            # players to related games, to avoid iterating each time
            # But, it's likely a small gain
            for game in games:
                if game["player_a_name"] == player_name:
                    opponent_name = game["player_b_name"]
                    outcome = game["outcome"]
                elif game["player_b_name"] == player_name:
                    opponent_name = game["player_a_name"]
                    outcome = 1 - game["outcome"]
                else:
                    continue

                opponent = self.players[opponent_name]
                opponent_ratings.append(opponent["rating"])
                opponent_uncertainties.append(opponent["rating_uncertainty"])
                outcomes.append(outcome)

            new_rating, new_rating_uncertainty, new_volatility = (
                compute_new_metrics_based_on_games(
                    player["rating"],
                    player["rating_uncertainty"],
                    player["volatility"],
                    opponent_ratings,
                    opponent_uncertainties,
                    outcomes,
                    tao=self.tao,
                    verbose=self.verbose,
                )
            )

            updated_players[player_name] = Glicko2Player(
                name=player_name,
                rating=new_rating,
                rating_uncertainty=new_rating_uncertainty,
                volatility=new_volatility,
            )

        self.players = updated_players

    def compute_winning_odds(
        self, player_a_name: str, player_b_name: str, **kwargs
    ) -> tuple[float, float, float]:
        # Always 0.0 for draw.

        player_b_confidence = compute_rating_confidence(
            self.players[player_b_name]["rating_uncertainty"]
        )
        winning_probability = compute_expected_outcome(
            self.players[player_a_name]["rating"],
            self.players[player_b_name]["rating"],
            player_b_confidence,
        )
        loosing_probability = 1 - winning_probability
        drawing_probability = 1 - winning_probability - loosing_probability

        if self.verbose:
            print(f"Winning probability: {100*winning_probability:.2f}")
            print(f"Loosing probability: {100*loosing_probability:.2f}")
            print(f"Drawing probability: {100*drawing_probability:.2f}")

        return winning_probability, loosing_probability, drawing_probability
