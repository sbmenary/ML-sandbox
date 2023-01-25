###
###  connect4.methods.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of miscellaneous methods for sharing between mutiple program.
"""


import logging, pickle, sys, time

import numpy as np

from connect4.utils import DebugLevel
from connect4.game  import BinaryPlayer, GameBoard, GameResult
from connect4.MCTS  import PolicyStrategy
from connect4.bot   import Bot_NeuralMCTS, Bot_VanillaMCTS


##  Global logger for this module
logger = logging.getLogger(__name__)



###======================###
###   Method defitions   ###
###======================###

def get_training_data_from_bot_game(model, duration:int=1, discount:float=1., num_random_moves:int=0, base_policy:PolicyStrategy=PolicyStrategy.NONE, noise_lvl:float=0.25) -> tuple :
    """
    Generate training data by playing a bot game against itself.

    Inputs:

        >  duration, int, default=1
           minimum time in seconds to run MCTS algorithm for

        >  discount, float, default=1.
           factor by which to multiply rewards with every step

        >  num_random_moves, int, default=0
           number of turns to use a uniformly random policy for at the start of the game

        >  base_policy, PolicyStrategy, default=NONE
           policy to apply to all moves after initial uniform random period, if NONE then fallback to class default

        >  noise_lvl, float [0,1], default=0.25
           fraction of noise to use when policy strategy is set to NOISY_POSTERIOR_POLICY

    Returns:

        >  np.ndarray shape [N, H, V, 1]: model inputs at each of N moves, from perspective of current player

        >  np.ndarray shape [N, H]: posterior policy at each of N moves

        >  np.ndarray shape [N, 1]: value of game result at each of N moves
    """
    ##  Create game and bot
    game_board = GameBoard()
    bot        = Bot_NeuralMCTS(model, noise_lvl=noise_lvl) if model else Bot_VanillaMCTS(noise_lvl=noise_lvl)

    ##  Log beginning of function
    logger.log(DebugLevel.VERY_LOW, f"[get_training_data_from_bot_game]  Created bot {bot} with noise level {noise_lvl:.3f}")
    if logger.isEnabledFor(DebugLevel.MEDIUM): logger.log(DebugLevel.MEDIUM, f"[get_training_data_from_bot_game]  Game board is:\n{game_board}")

    ##  Create containers for model input and output
    model_inputs, posteriors, values = [], [], []

    ##  Take moves until end of game, storing model input and target output at each turn
    turn_num, result, policy_strategy = 0, game_board.result, PolicyStrategy.UNIFORM_RANDOM
    logger.log(DebugLevel.VERY_LOW, f"[get_training_data_from_bot_game]  Beginning game with initial num_random_moves={num_random_moves}")
    while not result :

        ##  Update turn number, and decide whether to change from uniform random to base policy
        turn_num += 1
        if turn_num > num_random_moves : policy_strategy = base_policy
        logger.log(DebugLevel.MED_LOW, f"[get_training_data_from_bot_game]  Beginning turn number {turn_num} with duration={duration:.2f}, discount={discount:.2f}, policy_strategy={policy_strategy.name}")
        
        ##  Take move 
        bot.take_move(game_board, duration=duration, discount=discount, policy_strategy=policy_strategy)
        if logger.isEnabledFor(DebugLevel.MEDIUM): logger.log(DebugLevel.MEDIUM, f"[get_training_data_from_bot_game]  Updated game board is:\n{game_board}")

        ##  Resolve input to NN model with shape [H, V, 1]; if we used a NN model then this is already stored, otherwise we build it on-the-fly
        ##  - note that model input is always "from the player's perspective", which means we multiply board values by -1 if the current player is O
        if model : 
            model_input = bot.root_node.model_input
        else : 
            model_input = bot.root_node.game_board.board.reshape((game_board.horizontal_size, game_board.vertical_size, 1))
            if bot.root_node.player == BinaryPlayer.O : model_input = -model_input

        ##  Store model inputs, target posterior policy, and player value (+1 for X and -1 for O) 
        ##  -  At this stage we don't know the final game result, and will have to update values vector when we do
        posterior, value = bot.root_node.get_posterior_policy(), bot.root_node.player.value
        model_inputs.append(model_input)
        posteriors  .append(posterior)
        values      .append(value)

        ##  Log the 
        if logger.isEnabledFor(DebugLevel.MED_HIGH): 
            logger.log(DebugLevel.MED_HIGH, f"[get_training_data_from_bot_game]  Model input for turn {turn_num} is:\n{model_input[:,:,0]}")
            logger.log(DebugLevel.MED_HIGH, f"[get_training_data_from_bot_game]  Posterior for turn {turn_num} is: {'  '.join([f'{p:.2f}' for p in posterior])}")
            logger.log(DebugLevel.MED_HIGH, f"[get_training_data_from_bot_game]  Player value for turn {turn_num} is: {value}")

        ##  Retrieve current game result from game board so we know whether to terminate game
        result = game_board.result

    ##  Log game result
    logger.log(DebugLevel.VERY_LOW, f"[get_training_data_from_bot_game]  Game result is {result.name} with value {result.value} after {turn_num} turns")
        
    ##  Backpropagate the value of the game result to all preceeding moves
    ##  -  state value[i] = player value[i] * game result[i] * discount[i], where discount evolves geometrically in reverse time
    backprop_value = result.get_game_score_for_player(BinaryPlayer.X)
    for idx in range(len(values)) :
        values[-1-idx] *= backprop_value
        backprop_value *= discount
    if logger.isEnabledFor(DebugLevel.MED_LOW): logger.log(DebugLevel.MED_LOW, f"[get_training_data_from_bot_game]  Backpropagated values are: {'  '.join([f'{v:.3f}' for v in values])}")

    ##  Return model input and output containers as np arrays (values must be reshaped to match Tensor shape expected by Keras)
    model_inputs, posteriors, values = np.array(model_inputs), np.array(posteriors), np.array(values).reshape((len(values),1))
    return model_inputs, posteriors, values




def play_bot_game(model1, model2, duration:int=1, discount:float=1., bot1_policy:PolicyStrategy=PolicyStrategy.NONE, bot2_policy:PolicyStrategy=PolicyStrategy.NONE) -> dict :
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

        >  bot1_policy, PolicyStrategy, default=NONE
           policy for the bot assigned to model1

        >  bot2_policy, PolicyStrategy, default=NONE
           policy for the bot assigned to model2

    Returns:

        >  dict: result of the game along with which model was randomly assigned to players X and O
    """

    ##  Resolve bot policies
    if not bot1_policy : bot1_policy = PolicyStrategy.GREEDY_POSTERIOR_POLICY if model1 else PolicyStrategy.GREEDY_POSTERIOR_VALUE
    if not bot2_policy : bot2_policy = PolicyStrategy.GREEDY_POSTERIOR_POLICY if model2 else PolicyStrategy.GREEDY_POSTERIOR_VALUE

    ##  Create game and bots
    game_board = GameBoard()
    bot1       = Bot_NeuralMCTS(model1, policy_strategy=bot1_policy) if model1 else Bot_VanillaMCTS(policy_strategy=bot1_policy)
    bot2       = Bot_NeuralMCTS(model2, policy_strategy=bot2_policy) if model2 else Bot_VanillaMCTS(policy_strategy=bot2_policy)
    logger.log(DebugLevel.VERY_LOW, f"[play_bot_game]  Created bot1 {bot1} with policy={bot1_policy}")
    logger.log(DebugLevel.VERY_LOW, f"[play_bot_game]  Created bot2 {bot2} with policy={bot2_policy}")
    if logger.isEnabledFor(DebugLevel.MEDIUM): logger.log(DebugLevel.MEDIUM, f"[get_training_data_from_bot_game]  Game board is:\n{game_board}")
    
    ##  Randomly shuffle who goes first, and keep track of the result
    ret = {"model1":BinaryPlayer.X, "model2":BinaryPlayer.O}
    if np.random.choice([True, False]) :
        bot1, bot2 = bot2, bot1
        ret = {"model1":BinaryPlayer.O, "model2":BinaryPlayer.X}
    logger.log(DebugLevel.VERY_LOW, f"[play_bot_game]  model1 is player {ret['model1'].name}, model2 is player {ret['model2'].name}")

    ##  Take moves until end of game
    turn_num, result = 0, game_board.result
    while not result :
        ##  Select bot to play
        bot = bot1 if game_board.to_play == BinaryPlayer.X else bot2

        ##  Take move
        turn_num += 1
        logger.log(DebugLevel.MED_LOW, f"[play_bot_game]  Beginning turn {turn_num} with duration={duration:.2f}, discount={discount:.2f}, bot={bot}")
        bot.take_move(game_board, duration=duration, discount=discount)
        if logger.isEnabledFor(DebugLevel.MEDIUM): logger.log(DebugLevel.MEDIUM, f"[play_bot_game]  Updated game board is:\n{game_board}")

        ##  Update game result
        result = game_board.result

    ##  Log game result
    logger.log(DebugLevel.VERY_LOW, f"[play_bot_game]  Game result is {result.name} with value {result.value} after {turn_num} turns")
        
    ##  Add result to dictionary containing the player order, and return
    ret["result"] = result
    return ret
