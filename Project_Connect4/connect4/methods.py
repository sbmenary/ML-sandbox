###
###  connect4.methods.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of miscellaneous methods for sharing between mutiple program.
"""


import pickle, sys, time

import numpy as np

from connect4.utils import DebugLevel
from connect4.game  import BinaryPlayer, GameBoard, GameResult
from connect4.MCTS  import PolicyStrategy
from connect4.bot   import Bot_NeuralMCTS, Bot_VanillaMCTS



###======================###
###   Method defitions   ###
###======================###

def get_training_data_from_bot_game(model, duration:int=1, discount:float=1., random_first_move:bool=True, debug_lvl:DebugLevel=DebugLevel.MUTE) -> tuple :
    """
    Generate training data by playing a bot game against itself.

    Inputs:

        >  duration, int, default=1
           minimum time in seconds to run MCTS algorithm for

        >  discount, float, default=1.
           factor by which to multiply rewards with every step

        >  random_first_move, bool, default=True
           whether to play the first game move using a uniformly random policy

        >  debug_lvl, DebugLevel, default=MUTE
           level at which to print debug statements

    Returns:

        >  np.ndarray shape [N, H, V, 1]: model inputs at each of N moves, from perspective of current player

        >  np.ndarray shape [N, H]: posterior policy at each of N moves

        >  np.ndarray shape [N, 1]: value of game result at each of N moves
    """

    ##  Create game and bot
    game_board = GameBoard()
    bot        = Bot_NeuralMCTS(model) if model else Bot_VanillaMCTS()
    debug_lvl.message(DebugLevel.LOW, f"Using bot {bot}")
    debug_lvl.message(DebugLevel.LOW, game_board)

    ##  Create containers for model input and output
    model_inputs, posteriors, values = [], [], []

    ##  Take moves until end of game, storing model in and target out at each turn
    ##  -  values equal to +1 if the move is player X and -1 for player O
    ##  -  we do not invert sign of model_input because this already done by root_node
    ##  -  for first move, use a uniform random play strategy if configured
    result = game_board.get_result()
    policy_strategy = PolicyStrategy.UNIFORM_RANDOM if random_first_move else PolicyStrategy.NONE
    while not result :
        bot.take_move(game_board, duration=duration, discount=discount, policy_strategy=policy_strategy, debug_lvl=debug_lvl)
        debug_lvl.message(DebugLevel.LOW, game_board)
        if model : 
            model_input = bot.root_node.model_input
        else : 
            model_input = bot.root_node.game_board.board.reshape((game_board.horizontal_size, game_board.vertical_size, 1))
            if bot.root_node.player == BinaryPlayer.O : model_input = -model_input
        model_inputs.append(model_input)
        posteriors  .append(bot.root_node.get_posterior_policy())
        values      .append(bot.root_node.player.value)
        result = game_board.get_result()
        policy_strategy = PolicyStrategy.NONE
        
    ##  Backpropagate the value of the game result to all preceeding moves
    ##  -  at each turn, whether that player won/lost is taken account by the sign of the value multiplied by the sign of backprop_value
    backprop_value = result.get_game_score_for_player(BinaryPlayer.X)
    for idx in range(len(values)) :
        values[-1-idx] *= backprop_value
        backprop_value *= discount

    ##  Return containers as np arrays
    return np.array(model_inputs), np.array(posteriors), np.array(values).reshape((len(values),1))




def play_bot_game(model1, model2, duration:int=1, discount:float=1., debug_lvl:DebugLevel=DebugLevel.MUTE) -> dict :
    """
    Play a game between two bots and return the result

    Inputs:

        >  model1, tf.keras.Model
           neural network model for the player 1, if None then use a vanilla MCTS bot

        >  model2, tf.keras.Model
           neural network model for the player 1, if None then use a vanilla MCTS bot

        >  duration, int, default=1
           minimum time in seconds to run MCTS algorithm for

        >  discount, float, default=1.
           factor by which to multiply rewards with every step

        >  debug_lvl, DebugLevel, default=MUTE
           level at which to print debug statements

    Returns:

        >  dict: result of the game along with which model was randomly assigned to players X and O
    """

    ##  Create game and bots
    policy_strategy = PolicyStrategy.SAMPLE_POSTERIOR_POLICY
    game_board = GameBoard()
    bot1       = Bot_NeuralMCTS(model1, policy_strategy=policy_strategy) if model1 else Bot_VanillaMCTS(policy_strategy=policy_strategy)
    bot2       = Bot_NeuralMCTS(model2, policy_strategy=policy_strategy) if model2 else Bot_VanillaMCTS(policy_strategy=policy_strategy)
    debug_lvl.message(DebugLevel.LOW, f"Using bot1 {bot1}")
    debug_lvl.message(DebugLevel.LOW, f"Using bot2 {bot2}")
    debug_lvl.message(DebugLevel.LOW, game_board)
    
    ##  Randomly shuffle who goes first, and keep track of the result
    ret = {"model1":BinaryPlayer.X, "model2":BinaryPlayer.O}
    if np.random.choice([True, False]) :
        bot1, bot2 = bot2, bot1
        ret = {"model1":BinaryPlayer.O, "model2":BinaryPlayer.X}

    ##  Take moves until end of game
    result = game_board.get_result()
    while not result :
        bot = bot1 if game_board.to_play == BinaryPlayer.X else bot2
        bot.take_move(game_board, duration=duration, discount=discount, debug_lvl=debug_lvl)
        result = game_board.get_result()
        
    ##  Add result to dictionary containing the player order, and return
    ret["result"] = result
    return ret
