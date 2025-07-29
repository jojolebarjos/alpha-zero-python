# Note: the following takes as default the FIDE implementation of the Elo score

import math

from rating import RatingPlayer, Games, RatingSystem

FIDE_STARTING_RATING = 1000.0
FIDE_HIGH_LEVEL_THRESHOLD = 2400.0
FIDE_NEW_PLAYER_THRESHOLD = 30
FIDE_DEFAULT_K_FACTOR = 20.0
FIDE_HIGH_LEVEL_K_FACTOR = 10.0
FIDE_NEW_PLAYER_K_FACTOR = 40.0


def get_k_factor(
    player_rating: float, games_played: int, fixed_k: int | None = None
) -> float:
    # Use fixed_k to use always the same k-factor for all players
    if fixed_k is not None:
        return fixed_k

    # Else, Using system similar to FIDE's k-factor system
    # But, FIDE also considers the age of the player, and is not putting a direct
    # threshold on 2400, as it is rathe "has reached 2400 at some point"

    if games_played < FIDE_NEW_PLAYER_THRESHOLD:
        return FIDE_NEW_PLAYER_K_FACTOR
    if player_rating < FIDE_HIGH_LEVEL_THRESHOLD:
        return FIDE_DEFAULT_K_FACTOR
    return FIDE_HIGH_LEVEL_K_FACTOR


def compute_expected_outcome(
    player_a_rating: float, player_b_rating: float
) -> tuple[float, float]:

    diff = (player_b_rating - player_a_rating) / 400
    expected_outcome_a_inverse = 1 + math.pow(10, diff)
    expected_outcome_a = 1 / expected_outcome_a_inverse
    expected_outcome_b = 1 - expected_outcome_a

    return expected_outcome_a, expected_outcome_b


def compute_unweighted_rating_change(
    player_a_rating: float, player_b_rating: float, outcome_for_a: float
) -> tuple[float, float]:

    expected_outcome_a, expected_outcome_b = compute_expected_outcome(
        player_a_rating, player_b_rating
    )
    unweighted_rating_change_a = outcome_for_a - expected_outcome_a
    unweighted_rating_change_b = (1 - outcome_for_a) - expected_outcome_b

    return unweighted_rating_change_a, unweighted_rating_change_b


def compute_new_rating(
    player_rating: float,
    unweighted_rating_change: float,
    k_factor: float = FIDE_DEFAULT_K_FACTOR,
) -> float:

    return player_rating + k_factor * unweighted_rating_change


def compute_new_ratings_based_on_game(
    player_a_rating: float,
    player_b_rating: float,
    outcome_for_a: float,
    player_a_games_played: int,
    player_b_games_played: int,
    fixed_k: int | None = None,
) -> tuple[float, float]:

    unweighted_rating_change_a, unweighted_rating_change_b = (
        compute_unweighted_rating_change(
            player_a_rating, player_b_rating, outcome_for_a
        )
    )

    player_a_k_factor = get_k_factor(player_a_rating, player_a_games_played, fixed_k)
    player_b_k_factor = get_k_factor(player_b_rating, player_b_games_played, fixed_k)

    new_rating_a = compute_new_rating(
        player_a_rating, unweighted_rating_change_a, player_a_k_factor
    )
    new_rating_b = compute_new_rating(
        player_b_rating, unweighted_rating_change_b, player_b_k_factor
    )

    return new_rating_a, new_rating_b


class EloPlayer(RatingPlayer):
    games_played: int


class EloSystem(RatingSystem):

    def __init__(
        self,
        starting_rating: float = FIDE_STARTING_RATING,
        fixed_k: int | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(starting_rating=starting_rating, verbose=verbose)
        self.players: dict[str, EloPlayer]

        self.fixed_k = fixed_k

    def create_new_player(self, name: str) -> EloPlayer:
        return EloPlayer(name=name, rating=self.starting_rating, games_played=0)

    def create_player(self, name: str, rating: float, games_played: int) -> EloPlayer:
        return EloPlayer(name=name, rating=rating, games_played=games_played)

    def display_players(self) -> None:
        super().display_players()
        print("Number of played games:")
        for player_name, player in self.players.items():
            print(f"{player_name}: {player['games_played']}")
        print()

    def update_ratings(self, games: Games) -> None:
        # Reminder: For Elo, the order of the games does matter!
        # Players that didn't play in this session will keep their ratings unchanged

        self.check_players_in_games(games)

        for game in games:
            player_a = self.players[game["player_a_name"]]
            player_b = self.players[game["player_b_name"]]

            new_player_a_rating, new_player_b_rating = (
                compute_new_ratings_based_on_game(
                    player_a["rating"],
                    player_b["rating"],
                    game["outcome"],
                    player_a["games_played"],
                    player_b["games_played"],
                    fixed_k=self.fixed_k,
                )
            )

            if self.verbose:
                print(f"Updating rating for {player_a['name']} vs {player_b['name']}")
                print(f"Current rating: {player_a['rating']} vs {player_b['rating']}")
                print(
                    f"Nb games played: {player_a['games_played']} vs "
                    f"{player_b['games_played']}"
                )
                print(f"Outcome for {player_a['name']}: {game['outcome']}")
                print(
                    f"New rating: {new_player_a_rating:.2f} vs "
                    f"{new_player_b_rating:.2f}\n"
                )

            player_a["rating"] = new_player_a_rating
            player_b["rating"] = new_player_b_rating
            player_a["games_played"] += 1
            player_b["games_played"] += 1

    def compute_winning_odds(
        self, player_a_name: str, player_b_name: str, **kwargs
    ) -> tuple[float, float, float]:
        # Always 0.0 for draw

        winning_probability, loosing_probability = compute_expected_outcome(
            self.players[player_a_name]["rating"], self.players[player_b_name]["rating"]
        )
        drawing_probability = 1 - winning_probability - loosing_probability

        if self.verbose:
            print(f"Winning probability: {100*winning_probability:.2f}")
            print(f"Loosing probability: {100*loosing_probability:.2f}")
            print(f"Drawing probability: {100*drawing_probability:.2f}")

        return winning_probability, loosing_probability, drawing_probability
