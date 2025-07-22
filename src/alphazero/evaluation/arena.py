import numpy as np

from alphazero.data import Config, Episode, Match
from alphazero.predictor import Predictor


class Arena:
    """..."""

    def __init__(self, config: Config, predictors: dict[str, Predictor]) -> None:
        self.config = config
        self.predictors = predictors
        self.matches: list[Match] = []
        self.completed_matches: list[Match] = []

    def add_match(self, players: list[str]) -> None:
        for player in players:
            if player not in self.predictors:
                raise KeyError(player)
        state = self.config.sample_initial_state()
        episode = Episode([state], [], [])
        match = Match(players, state.reward, episode)
        self.matches.append(match)

    def step(self) -> bool:
        """..."""

        matches_per_predictor: dict[str, list[Match]] = {}

        for match in self.matches:
            assert match.episode is not None
            last_state = match.episode.states[-1]
            if last_state.has_ended:
                continue
            player = match.players[last_state.player]
            if player not in matches_per_predictor:
                matches_per_predictor[player] = []
            matches_per_predictor[player].append(match)

        if not matches_per_predictor:
            return False

        for player, matches in matches_per_predictor.items():
            predictor = self.predictors[player]

            # TODO limit batch size, handle in multiple batches if needed

            states = []
            for match in matches:
                assert match.episode is not None
                last_state = match.episode.states[-1]
                states.append(last_state)
            predictions = predictor.predict_many(states)

            for match, prediction in zip(matches, predictions):
                # TODO temperature?
                action_index = np.random.choice(len(prediction.actions), p=prediction.policy)
                action = prediction.actions[action_index]
                state = action.sample_next_state()
                assert match.episode is not None
                match.episode.predictions.append(prediction)
                match.episode.actions.append(action_index)
                match.episode.states.append(state)

        ongoing_matches: list[Match] = []
        for match in self.matches:
            assert match.episode is not None
            last_state = match.episode.states[-1]
            match.reward = last_state.reward
            if last_state.has_ended:
                self.completed_matches.append(match)
            else:
                ongoing_matches.append(match)

        self.matches = ongoing_matches

        return True
