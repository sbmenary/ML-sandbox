###
###  connect4.bot.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of bots that use MCTS to play connect 4.
"""

from __future__ import annotations
from abc  import ABC, abstractmethod
import logging

from connect4.utils import DebugLevel
from connect4.game  import GameBoard
from connect4.MCTS  import Node_Base, Node_NeuralMCTS, Node_VanillaMCTS, PolicyStrategy


##  Global logger for this module
logger = logging.getLogger(__name__)



###===============================###
###   Bot_Base class definition   ###
###===============================###

class Bot_Base(ABC) :
    
    def __init__(self, noise_lvl:float=0.25, policy_strategy:PolicyStrategy=PolicyStrategy.GREEDY_POSTERIOR_VALUE) :
        """
        Class Bot_Base

        Abstraction of common bot actions, such as running MCTS to choose a move from a current game board. 
        Does not specify MCTS node type. Derived class must over-ride the create_root_node method.
        """
        logger.log(DebugLevel.MED_LOW, f"[Bot_Base.__init__]  Creating bot with noise_lvl={noise_lvl:.3f}, policy_strategy={policy_strategy.name}")

        ##  Store a copy of the root node to allow it to be updated and queried.
        self.root_node       = None
        self.noise_lvl       = noise_lvl
        self.policy_strategy = policy_strategy
    
    
    @abstractmethod
    def create_root_node(self, game_board:GameBoard) -> Node_Base :
        """
        Create a root node for a given MCTS node type.
        """
        raise NotImplementedError()
        
        
    def choose_action(self, 
                      game_board:GameBoard = None, 
                      duration:int         = 1, 
                      max_sim_steps:int    = -1, 
                      discount             = 1.,
                      create_new_root_node = True,
                      policy_strategy      = PolicyStrategy.NONE, 
                      argv_choose_action   = []) -> int :
        """
        Perform a timed MCTS to choose a move.

        Inputs:

            >  game_board, GameBoard
               current game board

            >  duration, int, default=1
               minimum time in seconds to run MCTS algorithm for

            >  max_sim_steps, int, default=-1
               maximum number of turns per simulation, if -ve then no maximum

            >  discount, float, default=1.
               factor by which to multiply rewards with every step

            >  create_new_root_node, bool, default=True
               whether to replace the stored root node from previously run MCTS steps
               Note: create_new_root_node=False is ignored if no game_board object is provided

            >  policy_strategy, PolicyStrategy, default=NONE
               policy strategy by which to choose actions

            >  argv_choose_action, list, default=[]
               arguments to be passed on to node.choose_action

        Returns:

            >  int, column index of chosen action
        """
        if logger.isEnabledFor(DebugLevel.MEDIUM): 
            logger.log(DebugLevel.MEDIUM, f"[bot.choose_action]  Called with duration={duration:.2f}, max_sim_steps={max_sim_steps}, discount={discount:.2f}, create_new_root_node={create_new_root_node}, policy_strategy={policy_strategy.name}, argv_choose_action={argv_choose_action}")
        
        ##  If game has ended then cannot generate a new action
        if game_board and game_board.result :
            raise RuntimeError(f"Game is in terminal state {game_board.result}")

        ##  Create root_node from game_board provided
        ##  -  fall back to stored root_node if game_board is None
        if create_new_root_node and game_board :
            logger.log(DebugLevel.MEDIUM, f"[bot.choose_action]  Creating new root node")
            root_node = self.create_root_node(game_board)
        elif self.root_node : 
            logger.log(DebugLevel.MEDIUM, f"[bot.choose_action]  Using stored root node")
            root_node = self.root_node
        else :
            raise RuntimeError("Requested to use a previously stored root node but none available")

        ##  Call timed_MCTS to update tree values
        logger.log(DebugLevel.MEDIUM, f"[bot.choose_action]  Running timed MCTS from root node")
        root_node.timed_MCTS(duration      = duration     , 
                             max_sim_steps = max_sim_steps, 
                             discount      = discount)

        ##  Choose action, over-riding stored policy_strategy if a new one is provided
        if not policy_strategy :
            policy_strategy = self.policy_strategy
            logger.log(DebugLevel.MEDIUM, f"[bot.choose_action]  Updating policy_strategy to stored value: {policy_strategy.name}")

        logger.log(DebugLevel.MEDIUM, f"[bot.choose_action]  Choosing action with policy_strategy={policy_strategy.name}")
        action = root_node.choose_action(*argv_choose_action, policy_strategy=policy_strategy)

        ##  Print debug info
        if logger.isEnabledFor(DebugLevel.HIGHEST): 
            logger.log(DebugLevel.HIGHEST, f"[bot.choose_action]  Tree summary is:\n,{root_node.tree_summary()}")
        if logger.isEnabledFor(DebugLevel.MED_LOW): 
            logger.log(DebugLevel.MED_LOW, f"[bot.choose_action]  Action values are:  {'  '.join([f'{x.get_action_value():.3f}'.ljust(6) if x else 'N/A   ' for x in root_node.children])}")
            logger.log(DebugLevel.MED_LOW, f"[bot.choose_action]  Visit counts are:   {'  '.join([f'{x.num_visits}'.ljust(6) if x else 'N/A   ' for x in root_node.children])}")
            logger.log(DebugLevel.MED_LOW, f"[bot.choose_action]  Selecting action {action}")
        
        ##  Store root node
        self.root_node = root_node

        ##  Return best action from tree evaluation
        return action
    
    
    def num_itr(self) -> int :
        """
        Return number of MCTS iterations applied to the root node.
        """
        
        ##  If no root node created then answer is 0
        if not self.root_node :
            return 0
            
        ##  Otherwise query root node for num_visits
        return self.root_node.num_visits
    
    
    def take_move(self, 
                  game_board:GameBoard, 
                  duration:int         = 1, 
                  max_sim_steps:int    = -1, 
                  discount:float       = 1., 
                  create_new_root_node = True,
                  policy_strategy      = PolicyStrategy.NONE, 
                  argv_choose_action   = []) -> GameBoard :
        """
        Run MCTS to choose an action and apply it to the game board.

        Inputs:

            >  game_board, GameBoard
               current game board

            >  duration, int, default=1
               minimum time in seconds to run MCTS algorithm for

            >  max_sim_steps, int, default=-1
               maximum number of turns per simulation, if -ve then no maximum

            >  discount, float, default=1.
               factor by which to multiply rewards with every step

            >  create_new_root_node, bool, default=True
               whether to replace the stored root node from previously run MCTS steps

            >  argv_choose_action, list, default=[]
               arguments to be passed on to node.choose_action

        Returns:

            >  GameBoard, game board with bot action applied
        """

        ##  Use timed MCTS to obtain a bot action
        logger.log(DebugLevel.MED_LOW, f"[bot.take_move]  Calling self.choose_action")
        action = self.choose_action(game_board, duration, max_sim_steps, discount, create_new_root_node, policy_strategy, argv_choose_action)

        ##  Apply the bot move
        logger.log(DebugLevel.MED_LOW, f"[bot.take_move]  Applying action {action} to game board")
        game_board.apply_action(action)
        return game_board



###=====================================###
###   Bot_NeuralMCTS class definition   ###
###=====================================###

class Bot_NeuralMCTS(Bot_Base) :
    
    def __init__(self, model, c:float=1., noise_lvl:float=.25, policy_strategy:PolicyStrategy=PolicyStrategy.SAMPLE_POSTERIOR_POLICY) :
        super().__init__(noise_lvl=noise_lvl, policy_strategy=policy_strategy)
        logger.log(DebugLevel.MED_LOW, f"[Bot_NeuralMCTS.__init__]  Setting model={model.name}, c={c:.3f}")
        self.model     = model
        self.c         = c

    def create_root_node(self, game_board:GameBoard) -> Node_Base :
        """
        Create a Neural MCTS node.
        """
        return Node_NeuralMCTS(game_board, params=[self.model, self.c], noise_lvl=self.noise_lvl, policy_strategy=self.policy_strategy)



###======================================###
###   Bot_VanillaMCTS class definition   ###
###======================================###

class Bot_VanillaMCTS(Bot_Base) :
    
    def create_root_node(self, game_board:GameBoard) -> Node_Base :
        """
        Create a vanilla MCTS node.
        """
        return Node_VanillaMCTS(game_board, label="ROOT", noise_lvl=self.noise_lvl, policy_strategy=self.policy_strategy)
    
        