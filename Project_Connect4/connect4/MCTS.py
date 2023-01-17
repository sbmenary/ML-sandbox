###
###  connect4.MCTS.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of nodes and methods for Monte Carlo Tree Search.
"""

from __future__ import annotations
from abc        import ABC, abstractmethod
import time

import numpy as np

from connect4.utils import DebugLevel
from connect4.game  import BinaryPlayer, GameBoard, GameResult



###===============================###
###   Node_Base class definition   ###
###===============================###

class Node_Base(ABC) :
    
    def __init__(self, game_board:GameBoard, parent:Node=None, params:list=[], shallow_copy_board:bool=False, 
                 a_idx=-1, label=None, greedy=False) :
        """
        Class Node_Base
        
        - Used as part of MCTS algorithm. 
        - Stores total score and number of visits
        - Stores a list of children and a reference to the parent node
        - Provides methods for node selection, expansion, simulation, backpropagation
        - Abstract base class, requires derived class to implement expansion and simulation policies
        
        Inputs:
        
            > game_board, GameBoard
              state of the game at this node
              
            > parent, None, default=None
              reference to the parent node, only equals None if this is a root node
              
            > params, list, default=[]
              hyper-parameters to be used in models
              
            > shallow_copy_board, bool, default=False
              whether to only create a shallow copy of the game board - caution: improves memory efficiency 
              but may lead to undefined behaviour if either one of the referenced objects is updated
              
            > a_idx, int, default=-1
              which of parent's actions this node is derived from
              
            > label, str, default=None
              label for the node, used when generating summary strings
        """
                
        self.game_board  = game_board.deep_copy()
        self.actions     = game_board.get_available_actions()
        self.player      = game_board.to_play
        self.is_terminal = True if len(self.actions) == 0 else False
        self.children    = [None for a_idx in range(len(self.actions))]
        self.parent      = parent
        self.total_score = 0
        self.num_visits  = 0
        self.params      = params
        self.a_idx       = a_idx
        self.label       = label
        self.greedy      = greedy
        
        
    def __str__(self) -> str :
        """
        Return a string representation of the current node.
        """
        
        ##  Figure out parent / children info
        is_root              = False if self.parent else True
        num_children         = len(self.children)
        num_visited_children = len([c for c in self.children if c])
        
        ##  Begin str with node label if one provided
        ret  = f"[{self.label}] " if self.label else ""
        
        ##  Add some node information
        ret += f"N={self.num_visits}, T={self.total_score:.3f}, is_root={is_root}, is_leaf={self.is_terminal}"
        ret += f", num_children={num_children}, num_visited_children={num_visited_children}"
        
        ##  Return str
        return ret

        
    def choose_action(self, debug_lvl:DebugLevel=DebugLevel.MUTE, temperature:float=1.) -> int :
        """
        Choose an action by sampling over the posterior
        """
        
        ##  Select greedy action if requested
        if self.greedy :
        	debug_lvl.message(DebugLevel.LOW, f"Selecting greedy action")
        	return self.choose_greedy_action()

        ##  Otherwise sample from posterior distribution
        posterior = self.get_posterior_policy(temperature=temperature)
        debug_lvl.message(DebugLevel.LOW, f"Sampling action from posterior policy {' '.join([f'{x:.2f}' for x in posterior])}")
        return np.random.choice(len(posterior), p=posterior)
        
        
    def choose_greedy_action(self, debug_lvl:DebugLevel=DebugLevel.MUTE) -> int :
        """
        Return the optimal action using greedy policy based on the currently stored values.
        """
        
        ##  If this is a terminal node then no actions available
        if self.is_terminal : 
            return None
        
        ##  Find the index of the best child node, and return the corresponding action
        ##  - if no actions evaluated then argmax will return first action by default
        child_scores = [c.get_action_value() if c else -np.inf for c in self.children]
        best_a_idx   = np.argmax(child_scores)
        return self.actions[best_a_idx]
        
        
    def get_action_value(self) -> float :
        """
        Return node score.
        """
        
        ##  If node has not been visited then return -inf
        if self.num_visits == 0 :
            return -np.inf
        
        ##  Otherwise return mean reward per visit
        return self.total_score / self.num_visits
        
        
    @abstractmethod
    def get_expansion_score(self) -> float :
        """
        Returns the score used to expand nodes.
        """
        raise NotImplementedError()
    
    
    def get_posterior_policy(self, temperature:float=1.) -> np.ndarray :
        """
        Get the posterior policy for the current MCTS tree.
        Posterior proportional to N^(1./temperature) where N is visit count, unless move is forbidden,
            in which case it is forced to zero.
        """
        posterior = []
        for a_idx in range(self.game_board.horizontal_size) :
            if a_idx not in self.actions :
                posterior.append(0)
                continue
            child_idx  = self.actions.index(a_idx)
            child_node = self.children[child_idx]
            if child_node : N = child_node.num_visits
            else : N = 0
            posterior.append(N**(1./temperature))
        posterior = np.array(posterior)
        posterior = posterior / posterior.sum()
        return posterior


    def multi_step_MCTS(self, num_steps:int, max_sim_steps:int=-1, discount:float=1., debug_lvl:DebugLevel=DebugLevel.MUTE) :
        """
        Perform many MCTS iterations using self as the root node.
        """
        for idx in range(num_steps) :
            debug_lvl.message(DebugLevel.MEDIUM, f"Running MCTS step {idx}")
            self.one_step_MCTS(max_sim_steps=max_sim_steps, discount=discount, debug_lvl=debug_lvl)
            debug_lvl.message(DebugLevel.MEDIUM, f"")


    def one_step_MCTS(self, max_sim_steps:int=-1, discount:float=1., debug_lvl:DebugLevel=DebugLevel.MUTE) :
        """
        Perform a single MCTS iteration using self as the root node.
        """
        
        ##  Select and expand from the root node
        leaf_node = self.select_and_expand(recurse=True, debug_lvl=debug_lvl)
        
        ##  Simulate and backprop from the selected child
        leaf_node.simulate_and_backprop(max_sim_steps=max_sim_steps, discount=discount, debug_lvl=debug_lvl)
        
        ##  Print updated tree if debug level is HIGH
        debug_lvl.message(DebugLevel.HIGH, f"Updated tree is:\n{self.tree_summary()}")
        
        
    def select_and_expand(self, recurse:bool=False, debug_lvl:DebugLevel=DebugLevel.MUTE) -> Node :
        """
        Select from node children according to tree traversal policy. If next state is None then create a 
        new child and return this.
        
        Inputs:
        
            > recurse, bool, default=False
              whether to recursively iterate through tree until a new leaf node is found.
              
            > debug_lvl, DebugLevel, default=MUTE
              level at which to print debug statements to help understand algorithm behaviour.
        """
        
        ##  If leaf node then nothing to expand
        if self.is_terminal :
            debug_lvl.message(DebugLevel.MEDIUM, f"Leaf node found")
            return self
                
        ##  Uniformly randomly expand from un-visited children
        unvisited_children = [c_idx for c_idx, c in enumerate(self.children) if not c]
        if len(unvisited_children) > 0 :
            a_idx = np.random.choice(unvisited_children)
            new_game_board = self.game_board.deep_copy()
            node_label = f"{self.game_board.to_play.name}:{self.actions[a_idx]}"
            debug_lvl.message(DebugLevel.MEDIUM, f"Select unvisited action {node_label}")
            new_game_board.apply_action(self.actions[a_idx])
            self.children[a_idx] = self.__class__(new_game_board, parent=self, params=self.params, 
                                                  shallow_copy_board=True, a_idx=a_idx, label=node_label)
            return self.children[a_idx]
        
        ##  Otherwise best child is that with highest UCB score
        a_idx = np.argmax([c.get_expansion_score() for c in self.children])
        best_child = self.children[a_idx]
        debug_lvl.message(DebugLevel.MEDIUM, f"Select known action {self.game_board.to_play.name}:{self.actions[a_idx]}")
        
        ##  If recurse then also select_and_expand from the child node
        if recurse :
            debug_lvl.message(DebugLevel.MEDIUM, "... iterating to next level ...")
            return best_child.select_and_expand(recurse=recurse, debug_lvl=debug_lvl)
        
        ##  Otherwise return selected child
        return best_child
        
    
    @abstractmethod
    def simulate(self, max_sim_steps:int=-1, discount:float=1., debug_lvl:DebugLevel=DebugLevel.MUTE) -> float :
        """
        Simulate a game starting from this node.
        
        Inputs:
        
            > max_sim_steps, int, default=-1
              if positive then determines how many moves to play before declaring a drawn game
        
            > discount, int, default=1.
              factor by which to multiply the reward each turn, causing a preference for short-term rewards if discount<1.
              
            > debug_lvl, DebugLevel, default=MUTE
              level at which to print debug statements to help understand algorithm behaviour.
              
        Returns:
        
            > float
              the score of the simulation, defined as +1 is X wins, -1 if O wins, 0 for a draw
        """
        raise NotImplementedError()
    
    
    def simulate_and_backprop(self, max_sim_steps:int=-1, discount:float=1.,
                              debug_lvl:DebugLevel=DebugLevel.MUTE) -> float :
        """
        Simulate a game starting from this node. Backpropagate the resulting score up the whole tree.
        Result is the score from player X's perspective.
        
        Inputs:
        
            > max_sim_steps, int, default=-1
              if positive then determines how many moves to play before declaring a drawn game
              
            > debug_lvl, DebugLevel, default=MUTE
              level at which to print debug statements to help understand algorithm behaviour.
        """
        
        ##  Simulated game and obtain instance of GameResult
        result = self.simulate(max_sim_steps=max_sim_steps, discount=discount, debug_lvl=debug_lvl)
        
        ##  Update this node and backprop up the tree
        self.update_and_backprop(result, discount=discount, debug_lvl=debug_lvl)
            
            
    def timed_MCTS(self, duration:int, max_sim_steps:int=-1, discount:float=1., debug_lvl:DebugLevel=DebugLevel.MUTE) -> int :
        """
        Perform MCTS iterations with self as the root node until duration (in seconds) has elapsed.
        After this time, MCTS will finish its current iteration, so total execution time is > duration.
        """
        
        ##  Keep calling self.one_step_MCTS until required duration has elapsed
        start_time   = time.time()
        current_time = start_time
        num_itr = 0
        while current_time - start_time < duration :
            debug_lvl.message(DebugLevel.MEDIUM, f"Running MCTS step")
            self.one_step_MCTS(max_sim_steps=max_sim_steps, discount=discount, debug_lvl=debug_lvl)
            current_time = time.time()
            num_itr += 1
        return num_itr
        
        
    def tree_summary(self, indent_level:int=0) :
        """
        Return a multi-line str summarising every node in the tree.
        """
        
        ##  Summarise this node
        ret = ("     "*indent_level +
               f"> [{indent_level}{f': {self.label}' if self.label else ''}] N={self.num_visits}, T={self.total_score:.3f}, " +
               f"E={self.get_expansion_score():.3f}, Q={self.get_action_value():.3f}")
        
        ##  Recursively add the summary of each child node, iterating the indent level to reflect tree depth
        for a, c in zip(self.actions, self.children) :
            if c :
                ret += f"\n{c.tree_summary(indent_level+1)}"
            else :
                ret += "\n" + "     "*(indent_level+1) + "> None"
                
        ##  Return
        return ret
        
        
    def update(self, result:float, debug_lvl:DebugLevel=DebugLevel.MUTE) -> None :
        """
        Update the score and visit counts for this node.
        """
        
        ##  Resolve score for this node given the game result
        ##  - score is from the viewpoint of the parent, since this is the one deciding whether to come here!
        ##  - if no parent exists then this is a ROOT node, and we assign a score of 0. by default
        if self.parent :
        	if self.parent.player == BinaryPlayer.O :
        		result = -result
        else :
            result = 0.
        debug_lvl.message(DebugLevel.MEDIUM, 
              f"Node {self.label} with parent={self.parent.player.name if self.parent else 'NONE'}, N={self.num_visits}, T={self.total_score:.2f} receiving score {result:.2f}")
        
        ##  Update total score and number of visits for this node
        self.total_score += result
        self.num_visits  += 1
        
        
    def update_and_backprop(self, result:float, discount:float=1., 
                            debug_lvl:DebugLevel=DebugLevel.MUTE) -> None :
        """
        Update the score and visit counts for this node and backprop to all parents.
        """
        
        ##  Update this node
        self.update(result, debug_lvl=debug_lvl)
        
        ##  Recursively update all parent nodes
        if self.parent :
            self.parent.update_and_backprop(discount*result, debug_lvl=debug_lvl)

        

###======================================###
###   Node_NeuralMCTS class definition   ###
###======================================###

class Node_NeuralMCTS(Node_Base) :
    
    def __init__(self, game_board:GameBoard, parent:Node_Base=None, params:list=[], shallow_copy_board:bool=False, 
                 a_idx=-1, label=None, greedy=False) :
        """
        Class Node_NeuralMCTS
        
        - Used as part of MCTS algorithm. 
        - Stores total score and number of visits
        - Stores a list of children and a reference to the parent node
        - Provides methods for node selection, expansion, simulation, backpropagation
        - Derived from Node_Base, implements UCT expansion and one-step TD simulation with neural policy/value function
        
        Inputs:
        
            > game_board, GameBoard
              state of the game at this node
              
            > parent, None, default=None
              reference to the parent node, only equals None if this is a root node
              
            > params, list, default=[]
              hyper-parameters: [mode, c_UCT]
              
            > shallow_copy_board, bool, default=False
              whether to only create a shallow copy of the game board - caution: improves memory efficiency 
              but may lead to undefined behaviour if either one of the referenced objects is updated

            > a_idx, int, default=-1
              which of parent's actions this node is derived from
              
            > label, str, default=None
              label for the node, used when generating summary strings
        """

        ##  Call Node_Base initialiser with params=[model, c]
        super().__init__(game_board, parent, params, shallow_copy_board, a_idx, label, greedy)
        
        ##  Resolve the prior_prob for this node
        self.prior_prob = parent.child_priors[a_idx] if parent else 0
        
        ##  Resolve the hyper-params
        self.model   = params[0]
        self.c       = params[1]
        
        ##  Construct model input
        model_input = game_board.board
        model_input = model_input.reshape((1, model_input.shape[0], model_input.shape[1], 1))
        
        ##  Store model input and output
        self.model_input = model_input[0,:,:,:]
        self.child_priors, self.prior_value = self.model(model_input)
        self.child_priors, self.prior_value = self.child_priors.numpy()[0], self.prior_value.numpy()[0,0]
        
        
    def get_expansion_score(self) -> float :
        """
        Returns the UCT score of this node
        """
        
        ##  If node has no parent then no UCT score exists
        if not self.parent :
            return np.nan
        
        ##  Calculate mean score from past games
        mean_score = self.total_score / self.num_visits
        
        ##  Otherwise calculate UCT score
        return mean_score + self.c*self.prior_prob*np.sqrt(self.parent.num_visits) / (1+self.num_visits)
    
    
    def simulate(self, max_sim_steps:int=-1, discount:float=1., debug_lvl:DebugLevel=DebugLevel.MUTE) -> float :
        """
        Return value of a game from player X's perspective
        """
    
        ##  Check if game has already been won
        result = self.game_board.get_result()
        if result :
            debug_lvl.message(DebugLevel.MEDIUM, f"Leaf node found with result {result.name}")
            return result.get_game_score_for_player(BinaryPlayer.X)
                  
        ##  Otherwise return the NN value for this state, which is alway from player X's point of view
        debug_lvl.message(DebugLevel.MEDIUM, f"Simulation using prior value {self.prior_value:.4f}")
        return self.prior_value



###=======================================###
###   Node_VanillaMCTS class definition   ###
###=======================================###

class Node_VanillaMCTS(Node_Base) :
    
    def __init__(self, game_board:GameBoard, parent:Node_Base=None, params:list=[2.], shallow_copy_board:bool=False, 
                 a_idx=-1, label=None, greedy=False) :
        """
        Class Node_VanillaMCTS
        
        - Used as part of MCTS algorithm. 
        - Stores total score and number of visits
        - Stores a list of children and a reference to the parent node
        - Provides methods for node selection, expansion, simulation, backpropagation
        - Derived from Node_Base, implements vanilla UCB1 expansion and uniform-random simulation
        
        Inputs:
        
            > game_board, GameBoard
              state of the game at this node
              
            > parent, None, default=None
              reference to the parent node, only equals None if this is a root node
              
            > params, list, default=[2.]
              hyper-parameter controlling strength of exploration in UCB search
              
            > shallow_copy_board, bool, default=False
              whether to only create a shallow copy of the game board - caution: improves memory efficiency 
              but may lead to undefined behaviour if either one of the referenced objects is updated

            > a_idx, int, default=-1
              which of parent's actions this node is derived from
              
            > label, str, default=None
              label for the node, used when generating summary strings
        """
        
        ##  Call Node_Base initialiser with params=[UCB_c]
        super().__init__(game_board, parent, params, shallow_copy_board, a_idx, label, greedy)
        
        
    def get_expansion_score(self) -> float :
        """
        Returns the UCB score of this node
        """
        
        ##  Retreive value of UCB exploration-strength hyper-param
        UCB_c = self.params[0]
                
        ##  If node is un-visited then the UCB score is infinite
        if UCB_c != 0 and self.num_visits == 0 :
            return np.inf
        
        ##  If node has no parent then no UCB score exists
        if not self.parent :
            return np.nan
        
        ##  Calculate mean score from past games
        mean_score = self.total_score / self.num_visits
        
        ##  Otherwise calculate UCB score
        return mean_score + UCB_c * np.sqrt(np.log(self.parent.num_visits) / self.num_visits)
        
    
    def simulate(self, max_sim_steps:int=-1, discount:float=1., debug_lvl:DebugLevel=DebugLevel.MUTE) -> float :
        """
        Simulate a game starting from this node.
        Assumes that both players act according to a uniform-random policy.
        
        Inputs:
        
            > max_sim_steps, int, default=-1
              if positive then determines how many moves to play before declaring a drawn game
              
            > debug_lvl, DebugLevel, default=MUTE
              level at which to print debug statements to help understand algorithm behaviour.
              
        Returns:
        
            > float
              the score of the simulation, defined as +1 is X wins, -1 if O wins, 0 for a draw
        """
        
        ##  Check if game has already been won
        ##  - if so then return score
        ##  - score is -1 if target player has lost, +1 if they've won, and 0 for a draw
        result = self.game_board.get_result()
        if result :
            debug_lvl.message(DebugLevel.MEDIUM, f"Leaf node found with result {result.name}")
            return result.get_game_score_for_player(BinaryPlayer.X)
                
        ##  Create copy of game board to play simulation
        simulated_game = self.game_board.deep_copy()
        
        ##  Keep playing moves until one of terminating conditions is reached:
        ##  1. game is won by a player
        ##  2. no further moves are possible, game is considered a draw
        ##  3. maximum move limit is reached, game is considered a draw
        turn_idx, is_terminal, compound_discount, result = 0, False, 1., GameResult.NONE
        trajectory = []
        while not is_terminal :
            turn_idx += 1
            compound_discount *= discount
            action = np.random.choice(simulated_game.get_unfilled_columns())
            trajectory.append(f"{simulated_game.to_play.name}:{action}")
            simulated_game.apply_action(action)
            result = simulated_game.get_result()
            if result :
                is_terminal = True
                  
        ##  Debug trajectory
        debug_lvl.message(DebugLevel.MEDIUM, f"Simulation ended with result {result.name} with compound_discount={compound_discount:.3f}")
        debug_lvl.message(DebugLevel.MEDIUM, f"Simulated trajectory was: {' '.join(trajectory)}")
                                
        ##  Return score
        return compound_discount*result.get_game_score_for_player(BinaryPlayer.X)