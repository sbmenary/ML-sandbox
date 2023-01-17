###
###  connect4.bot.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of bots that use MCTS to play connect 4.
"""

from __future__ import annotations
from abc  import ABC, abstractmethod

from connect4.utils import DebugLevel
from connect4.game  import GameBoard
from connect4.MCTS  import Node_Base, Node_NeuralMCTS, Node_VanillaMCTS



###===============================###
###   Bot_Base class definition   ###
###===============================###

class Bot_Base(ABC) :
    
    def __init__(self, greedy=False) :
        """
        Class Bot_Base

        Abstraction of common bot actions, such as running MCTS to choose a move from a current game board. 
        Does not specify MCTS node type. Derived class must over-ride the create_root_node method.
        """

        ##  Store a copy of the root node to allow it to be updated and queried.
        self.root_node = None
        self.greedy    = greedy
    
    
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
                      debug_lvl:DebugLevel = DebugLevel.MUTE,
                      *argv_choose_action) -> int :
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

            >  debug_lvl, DebugLevel, default=MUTE
               level at which to print debug statements

            > argv_choose_action, *list, default=[]
              arguments to be passed on to node.choose_action

        Returns:

            >  int, column index of chosen action
        """
        
        ##  If game has ended then cannot generate a new action
        game_result = game_board.get_result() 
        if game_board and game_board.get_result() :
            raise RuntimeError(f"Game is in terminal state {game_result}")

        ##  Create root_node from game_board provided
        ##  -  fall back to stored root_node if game_board is None
        if create_new_root_node and game_board :
            root_node = self.create_root_node(game_board)
        elif self.root_node : 
            root_node = self.root_node
        else :
            raise RuntimeError("Requested to use a previously stored root node but none available")

        ##  Call timed_MCTS to update tree values 
        root_node.timed_MCTS(duration      = duration     , 
                             max_sim_steps = max_sim_steps, 
                             discount      = discount,
                             debug_lvl     = debug_lvl    )
        action = root_node.choose_action(debug_lvl=debug_lvl, *argv_choose_action)

        ##  Print debug info
        debug_lvl.message(DebugLevel.HIGH, root_node.tree_summary())
        debug_lvl.message(DebugLevel.LOW, 
              "Action values are:  " + "  ".join([f"{x.get_action_value():.3f}".ljust(6) if x else "N/A   " for x in root_node.children]))
        debug_lvl.message(DebugLevel.LOW, 
              "Visit counts are:   " + "  ".join([f"{x.num_visits}".ljust(6) if x else "N/A   " for x in root_node.children]))
        debug_lvl.message(DebugLevel.LOW, 
              f"Selecting action {action}")
        
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
                  debug_lvl:DebugLevel = DebugLevel.MUTE) -> GameBoard :
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

            >  debug_lvl, DebugLevel, default=MUTE
               level at which to print debug statements

        Returns:

            >  GameBoard, game board with bot action applied
        """

        ##  Use timed MCTS to obtain a bot action
        action = self.choose_action(game_board, duration, max_sim_steps, discount, create_new_root_node, debug_lvl)

        ##  Apply the bot move
        game_board.apply_action(action)
        return game_board



###=====================================###
###   Bot_NeuralMCTS class definition   ###
###=====================================###

class Bot_NeuralMCTS(Bot_Base) :
    
    def __init__(self, model, c=1., greedy=False) :
        super().__init__(greedy=greedy)
        self.model  = model
        self.c      = c

    def create_root_node(self, game_board) :
        """
        Create a Neural MCTS node.
        """
        return Node_NeuralMCTS(game_board, params=[self.model, self.c], greedy=self.greedy)



###======================================###
###   Bot_VanillaMCTS class definition   ###
###======================================###

class Bot_VanillaMCTS(Bot_Base) :
    
    def create_root_node(self, game_board) :
        """
        Create a vanilla MCTS node.
        """
        return Node_VanillaMCTS(game_board, label="ROOT", greedy=self.greedy)
    
        