import math

from rating import RatingPlayer
from match_making import MatchMaking

NON_DIVISION_PER_ZERO_BIAS = 1e-10


class RatingMatchMaking(MatchMaking):
    def weight_function(self, player: RatingPlayer, opponent: RatingPlayer) -> float:

        weight = 1 / (
            math.pow(player["rating"] - opponent["rating"], 2)
            + NON_DIVISION_PER_ZERO_BIAS
        )

        return weight

    def opponent_printing_function(
        self, player: RatingPlayer, opponent: RatingPlayer, weight: float
    ) -> str:
        text = f"{opponent['name']} (rating: {opponent['rating']}, weight: {weight})"
        return text


class WinningProbabilityMatchMaking(MatchMaking):
    def weight_function(self, player: RatingPlayer, opponent: RatingPlayer) -> float:

        win_p, lose_p, draw_p = self.rating_system.compute_winning_odds(
            player["name"], opponent["name"]
        )
        weight = 1 / (math.pow(win_p - lose_p, 2) + NON_DIVISION_PER_ZERO_BIAS)

        return weight

    def opponent_printing_function(
        self, player: RatingPlayer, opponent: RatingPlayer, weight: float
    ) -> str:
        win_p, lose_p, draw_p = self.rating_system.compute_winning_odds(
            player["name"], opponent["name"]
        )
        text = (
            f"{opponent['name']} (rating: {opponent['rating']}, "
            f"winning probability: {win_p:.2f}, loosing probability: {lose_p:.2f}, "
            f"drawing probability: {draw_p:.2f}, weight: {weight})"
        )
        return text
