from pypokerengine.api.game import setup_config, start_poker
from heuristicAI import HeuristicPlayer
from consoleAI import ConsolePlayer 

init_def_prob = [
    [0.6, 0.2, 0.0, 0.2],
    [0.4, 0.4, 0.1, 0.1],
    [0.1, 0.7, 0.2, 0.0],
    [0.0, 0.6, 0.4, 0.0],
    [0.0, 0.3, 0.7, 0.0]
]

config = setup_config(max_round=10, initial_stack=200, small_blind_amount=1)
config.register_player(name="AI_1", algorithm=HeuristicPlayer(init_def_prob))
config.register_player(name="AI_2", algorithm=HeuristicPlayer(init_def_prob))
config.register_player(name="AI_3", algorithm=HeuristicPlayer(init_def_prob))
# config.register_player(name="Human", algorithm=ConsolePlayer())
game_result = start_poker(config, verbose=1)
print(game_result)