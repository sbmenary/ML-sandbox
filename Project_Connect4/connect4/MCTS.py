###
###  connect4.MCTS.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of nodes and methods for Monte Carlo Tree Search.
"""

from __future__ import annotations
from enum       import IntEnum
from abc        import ABC, abstractmethod
import logging, time

import numpy as np

from connect4.utils import DebugLevel
from connect4.game  import BinaryPlayer, GameBoard, GameResult


##  Global logger for this module
logger = logging.getLogger(__name__)



###=====================================###
###   PolicyStrategy class definition   ###
###=====================================###

class PolicyStrategy(IntEnum) :
    """
    An enumeration for the strategy for choosing actions.
    """
    NONE                    = 0
    UNIFORM_RANDOM          = 1
    GREEDY_PRIOR_VALUE      = 2
    GREEDY_POSTERIOR_VALUE  = 3
    GREEDY_PRIOR_POLICY     = 4
    GREEDY_POSTERIOR_POLICY = 5
    SAMPLE_PRIOR_POLICY     = 6
    SAMPLE_POSTERIOR_POLICY = 7
    NOISY_POSTERIOR_POLICY  = 8



###================================###
###   Node_Base class definition   ###
###================================###

class Node_Base(ABC) :
    
    def __init__(self, game_board:GameBoard, parent:Node=None, params:list=[], shallow_copy_board:bool=False, 
                 action=-1, label=None, noise_lvl:float=0.25, policy_strategy=PolicyStrategy.GREEDY_POSTERIOR_VALUE) :
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
              
            > action, int, default=-1
              which of parent's actions this node is derived from
              
            > label, str, default=None
              label for the node, used when generating summary strings

            > noise_lvl, float [0,1], default=0.25
              fraction of noise to use when policy strategy is set to NOISY_POSTERIOR_POLICY
              
            > policy_strategy, PolicyStrategy, default=PolicyStrategy.GREEDY_POSTERIOR_VALUE
              strategy for choosing actions
        """
        self.game_board      = game_board if shallow_copy_board else game_board.deep_copy()
        self.actions         = game_board.get_available_actions()
        self.player          = game_board.to_play
        self.is_terminal     = True if game_board.result else False
        self.children        = [None for a_idx in range(len(self.actions))]
        self.parent          = parent
        self.action          = action
        self.total_score     = 0
        self.num_visits      = 0
        self.params          = params
        self.label           = label
        self.noise_lvl       = noise_lvl
        self.policy_strategy = policy_strategy
        logger.log(DebugLevel.VERY_HIGH, f"[Node_Base.__init__]  Created node with label='{label}', actions={self.actions}, is_terminal={self.is_terminal}, player={self.player.name}, parent={parent.label if parent else None}, action={action}, params={params}, noise_lvl={noise_lvl:.2f}, policy_strategy={policy_strategy.name}")
        
        
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


    def _select_action_greedy_prior_policy(self) -> int :
        raise NotImplementedError()


    def _select_action_greedy_prior_value(self) -> int :
        raise NotImplementedError()


    def _select_action_sample_prior_policy(self) -> int :
        raise NotImplementedError()

        
    def choose_action(self, temperature:float=1., policy_strategy=PolicyStrategy.NONE) -> int :
        """
        Choose an action according to the policy_strategy, falling back to stored value if NONE provided
        """
        logger.log(DebugLevel.HIGH, f"[Node_Base.choose_action]  Choosing action with temperature={temperature:.2f}, policy_strategy={policy_strategy.name}")

        ##  Make sure a policy strategy is set
        if not policy_strategy :
            if not self.policy_strategy :
                raise RuntimeError("No policy strategy set")
            policy_strategy = self.policy_strategy
            logger.log(DebugLevel.HIGH, f"[Node_Base.choose_action]  Setting policy_strategy to fallback value {policy_strategy.name}")
        
        ##  If this is a terminal node then no actions available
        if self.is_terminal : 
            logger.log(DebugLevel.HIGH, "[Node_Base.choose_action]  Node is terminal, no action to be selected, returning None")
            return None

        ##  Resolve policy strategy UNIFORM_RANDOM
        if policy_strategy == PolicyStrategy.UNIFORM_RANDOM :
            logger.log(DebugLevel.HIGH, f"[Node_Base.choose_action]  Selecting uniformly random action from {self.actions}")
            return np.random.choice(self.actions)
        
        ##  Resolve policy strategy GREEDY_PRIOR_VALUE
        if policy_strategy == PolicyStrategy.GREEDY_PRIOR_VALUE :
            logger.log(DebugLevel.HIGH, "[Node_Base.choose_action]  Selecting greedy action from prior values")
            return self._select_action_greedy_prior_value()
        
        ##  Resolve policy strategy GREEDY_POSTERIOR_VALUE
        if policy_strategy == PolicyStrategy.GREEDY_POSTERIOR_VALUE :
            logger.log(DebugLevel.HIGH, "[Node_Base.choose_action]  Selecting greedy action from posterior values")
            child_scores = [c.get_action_value() if c else -np.inf for c in self.children]
            best_a_idx   = np.argmax(child_scores)
            return self.actions[best_a_idx]
        
        ##  Resolve policy strategy GREEDY_PRIOR_POLICY
        if policy_strategy == PolicyStrategy.GREEDY_PRIOR_POLICY :
            logger.log(DebugLevel.HIGH, "[Node_Base.choose_action]  Selecting greedy action from prior policy")
            return self._select_action_greedy_prior_policy()

        ##  Resolve policy strategy GREEDY_POSTERIOR_POLICY
        if policy_strategy == PolicyStrategy.GREEDY_POSTERIOR_POLICY :
            posterior = self.get_posterior_policy(temperature=temperature)
            logger.log(DebugLevel.HIGH, f"[Node_Base.choose_action]  Selecting greedy action from posterior policy {' '.join([f'{x:.2f}' for x in posterior])}")
            return np.argmax(posterior)
        
        ##  Resolve policy strategy SAMPLE_PRIOR_VALUE
        if policy_strategy == PolicyStrategy.SAMPLE_PRIOR_POLICY :
            logger.log(DebugLevel.HIGH, "[Node_Base.choose_action]  Sampling action from prior policy")
            return self._select_action_sample_prior_policy()

        ##  Resolve policy strategy SAMPLE_POSTERIOR_POLICY
        if policy_strategy == PolicyStrategy.SAMPLE_POSTERIOR_POLICY :
            posterior = self.get_posterior_policy(temperature=temperature)
            logger.log(DebugLevel.HIGH, f"[Node_Base.choose_action]  Sampling action from posterior policy {' '.join([f'{x:.2f}' for x in posterior])}")
            return np.random.choice(len(posterior), p=posterior)

        ##  Resolve policy strategy NOISY_POSTERIOR_POLICY
        if policy_strategy == PolicyStrategy.NOISY_POSTERIOR_POLICY :
            posterior = self.get_posterior_policy(temperature=temperature)
            logger.log(DebugLevel.HIGH, f"Adding {100.*self.noise_lvl:.1f}% noise to posterior policy {' '.join([f'{x:.2f}' for x in posterior])}")
            noisy_dist = []
            for action in range(len(posterior)) :
                if action in self.actions : 
                    noisy_dist.append(1.)
                else :
                    noisy_dist.append(0.)
            noisy_dist = np.array(noisy_dist) / np.sum(noisy_dist)
            noisy_dist = (1-self.noise_lvl)*posterior + self.noise_lvl*noisy_dist
            logger.log(DebugLevel.HIGH, f"[Node_Base.choose_action]  Sampling action from noisy policy {' '.join([f'{x:.2f}' for x in noisy_dist])}")
            return np.random.choice(len(noisy_dist), p=noisy_dist)

        ##  If here then policy strategy not recognised!
        logger.error(f"[Node_Base.choose_action]  No policy implemented for strategy {policy_strategy.name}")
        raise NotImplementedError(f"No policy implemented for strategy {policy_strategy.name}")

        
        
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
        if logger.isEnabledFor(DebugLevel.HIGHEST): 
            logger.log(DebugLevel.HIGHEST, f"[Node_Base.get_posterior_policy]  Posterior {' '.join([f'{x:.2f}' for x in posterior])} constructed from actions {self.actions} with temperature {temperature:.2f}")
        return posterior


    def multi_step_MCTS(self, num_steps:int, max_sim_steps:int=-1, discount:float=1.) -> None :
        """
        Perform many MCTS iterations using self as the root node.
        """
        logger.log(DebugLevel.HIGH, f"[Node_Base.multi_step_MCTS]  Running {num_steps} MCTS iterations with max_sim_steps={max_sim_steps}, discount={discount:.2f}")
        for idx in range(num_steps) :
            self.one_step_MCTS(max_sim_steps=max_sim_steps, discount=discount)


    def one_step_MCTS(self, max_sim_steps:int=-1, discount:float=1.) -> None :
        """
        Perform a single MCTS iteration using self as the root node.
        """
        logger.log(DebugLevel.VERY_HIGH, "[Node_Base.one_step_MCTS]  Running MCTS iteration")
        
        ##  Select and expand from the root node
        leaf_node = self.select_and_expand(recurse=True)
        
        ##  Simulate and backprop from the selected child
        leaf_node.simulate_and_backprop(max_sim_steps=max_sim_steps, discount=discount)
        
        ##  Print updated tree if debug level is HIGHEST
        if logger.isEnabledFor(DebugLevel.HIGHEST): 
            logger.log(DebugLevel.HIGHEST, f"Updated tree is:\n{self.tree_summary()}")
        
        
    def select_and_expand(self, recurse:bool=False) -> Node :
        """
        Select from node children according to tree traversal policy. If next state is None then create a 
        new child and return this.
        
        Inputs:
        
            > recurse, bool, default=False
              whether to recursively iterate through tree until a new leaf node is found.
        """
        
        ##  If leaf node then nothing to expand
        if self.is_terminal :
            logger.log(DebugLevel.VERY_HIGH, f"[Node_Base.select_and_expand]  Leaf node found")
            return self
                
        ##  Uniformly randomly expand from un-visited children
        unvisited_children = [c_idx for c_idx, c in enumerate(self.children) if not c]
        if len(unvisited_children) > 0 :
            a_idx  = np.random.choice(unvisited_children)
            action = self.actions[a_idx]
            new_game_board = self.game_board.deep_copy()
            node_label = f"{self.game_board.to_play.name}:{action}"
            logger.log(DebugLevel.VERY_HIGH, f"[Node_Base.select_and_expand]  Selecting unvisited action {node_label}")
            new_game_board.apply_action(action)
            self.children[a_idx] = self.__class__(new_game_board, parent=self, params=self.params, shallow_copy_board=True, 
                                                  action=action, label=node_label, noise_lvl=self.noise_lvl)
            return self.children[a_idx]
        
        ##  Otherwise best child is that with highest UCB score
        a_idx = np.argmax([c.get_expansion_score() for c in self.children])
        action = self.actions[a_idx]
        best_child = self.children[a_idx]
        logger.log(DebugLevel.VERY_HIGH, f"[Node_Base.select_and_expand]  Select known action {self.game_board.to_play.name}:{action}")
        
        ##  If recurse then also select_and_expand from the child node
        if recurse :
            logger.log(DebugLevel.VERY_HIGH, f"[Node_Base.select_and_expand]  ...iterating to next level...")
            return best_child.select_and_expand(recurse=recurse)
        
        ##  Otherwise return selected child
        return best_child
        
    
    @abstractmethod
    def simulate(self, max_sim_steps:int=-1, discount:float=1.) -> float :
        """
        Simulate a game starting from this node.
        
        Inputs:
        
            > max_sim_steps, int, default=-1
              if positive then determines how many moves to play before declaring a drawn game
        
            > discount, int, default=1.
              factor by which to multiply the reward each turn, causing a preference for short-term rewards if discount<1.
              
        Returns:
        
            > float
              the score of the simulation, defined as +1 is X wins, -1 if O wins, 0 for a draw
        """
        raise NotImplementedError()
    
    
    def simulate_and_backprop(self, max_sim_steps:int=-1, discount:float=1.) -> float :
        """
        Simulate a game starting from this node. Backpropagate the resulting score up the whole tree.
        Result is the score from player X's perspective.
        
        Inputs:
        
            > max_sim_steps, int, default=-1
              if positive then determines how many moves to play before declaring a drawn game
        """
        
        ##  Simulated game and obtain instance of GameResult
        result = self.simulate(max_sim_steps=max_sim_steps, discount=discount)
        
        ##  Update this node and backprop up the tree
        self.update_and_backprop(result, discount=discount)
            
            
    def timed_MCTS(self, duration:int, max_sim_steps:int=-1, discount:float=1.) -> int :
        """
        Perform MCTS iterations with self as the root node until duration (in seconds) has elapsed.
        After this time, MCTS will finish its current iteration, so total execution time is > duration.
        """
        logger.log(DebugLevel.HIGH, f"[Node_Base.timed_MCTS]  Running MCTS steps for duration={duration:.2f} with max_sim_steps={max_sim_steps}, discount={discount:.2f}")
        
        ##  Keep calling self.one_step_MCTS until required duration has elapsed
        start_time   = time.time()
        current_time = start_time
        num_itr = 0
        while current_time - start_time < duration :
            self.one_step_MCTS(max_sim_steps=max_sim_steps, discount=discount)
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
               f"E={self.get_expansion_score():.3f}, Q={self.get_action_value():.3f}, {len(self.children)} children")
        
        ##  Recursively add the summary of each child node, iterating the indent level to reflect tree depth
        for a, c in zip(self.actions, self.children) :
            if not c : continue
            ret += f"\n{c.tree_summary(indent_level+1)}"
                
        ##  Return
        return ret
        
        
    def update(self, result:float) -> None :
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
        logger.log(DebugLevel.VERY_HIGH, f"[Node_Base.update]  Node {self.label} with parent={self.parent.player.name if self.parent else 'NONE'}, N={self.num_visits}, T={self.total_score:.2f} receiving score {result:.2f}")
        
        ##  Update total score and number of visits for this node
        self.total_score += result
        self.num_visits  += 1
        
        
    def update_and_backprop(self, result:float, discount:float=1.) -> None :
        """
        Update the score and visit counts for this node and backprop to all parents.
        """
        
        ##  Update this node
        self.update(result)
        
        ##  Recursively update all parent nodes
        if self.parent :
            self.parent.update_and_backprop(discount*result)

        

###======================================###
###   Node_NeuralMCTS class definition   ###
###======================================###

class Node_NeuralMCTS(Node_Base) :
    
    def __init__(self, game_board:GameBoard, parent:Node_Base=None, params:list=[], shallow_copy_board:bool=False, 
                 action:int=-1, label:str=None, noise_lvl:float=0.25, policy_strategy:PolicyStrategy=PolicyStrategy.GREEDY_POSTERIOR_VALUE) :
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

            > noise_lvl, float [0,1], default=0.25
              fraction of noise to use when policy strategy is set to NOISY_POSTERIOR_POLICY
              
            > policy_strategy, PolicyStrategy, default=PolicyStrategy.SAMPLE_POSTERIOR_POLICY
              strategy for choosing actions
        """

        ##  Call Node_Base initialiser with params=[model, c]
        super().__init__(game_board, parent, params, shallow_copy_board, action, label, noise_lvl, policy_strategy)
        
        ##  Resolve the prior_prob for this node
        self.prior_prob = parent.child_priors[action] if parent else 0
        
        ##  Resolve the hyper-params
        self.model = params[0]
        self.c     = params[1]
        
        ##  Construct model input
        ##  -  if this node is player O, multiply model_input by -1, so it becomes "from current player's perspective"
        model_input = game_board.board.copy()
        model_input = model_input.reshape((1, model_input.shape[0], model_input.shape[1], 1))
        if self.player == BinaryPlayer.O : model_input = -model_input
        
        ##  Store model input and output
        ##  -  if this node is player O, multiply prior_value by -1, so it becomes "from player X's perspective"
        ##  -  N.B. this is true because model input is from our own perspective
        self.model_input = model_input[0,:,:,:]
        self.child_priors, self.prior_value = self.model(model_input)
        self.child_priors, self.prior_value = self.child_priors.numpy()[0], self.prior_value.numpy()[0,0]
        if self.player == BinaryPlayer.O : self.prior_value = -self.prior_value

        ##  Log this node
        logger.log(DebugLevel.VERY_HIGH, f"[Node_NeuralMCTS.__init__]  Created node with model={self.model.name}, c={self.c:.3f}, prior_prob={prior_prob:.3f}, prior_value={self.prior_value:.3f}, child_priors={' '.join([f'{x:.2f}' for x in self.child_priors])}")


    def _select_action_greedy_prior_policy(self) -> int :
        prior = self.get_prior_policy()
        logger.log(DebugLevel.HIGH, f"[Node_NeuralMCTS._select_action_greedy_prior_policy]  Prior policy is {' '.join([f'{x:.2f}' for x in prior])}")
        return np.argmax(prior)


    def _select_action_greedy_prior_value(self) -> int :
        actions, values = [], []
        for a_idx, action in enumerate(self.actions) :
            child_node = self.children[a_idx]
            if not child_node :
                new_game_board = self.game_board.deep_copy()
                new_game_board.apply_action(action)
                child_node = self.__class__(new_game_board, parent=self, params=self.params, shallow_copy_board=True, action=action)
            v = child_node.prior_value
            actions.append(action)
            values .append(v)
        values = np.array(values)
        if self.player == BinaryPlayer.O :
            values = -values
        logger.log(DebugLevel.HIGH, f"[Node_NeuralMCTS._select_action_greedy_prior_value]  Prior values are {' '.join([f'{x:.2f}' for x in prior])}")
        return actions[np.argmax(values)]


    def _select_action_sample_prior_policy(self) -> int :
        prior = self.get_prior_policy()
        logger.log(DebugLevel.HIGH, f"[Node_NeuralMCTS._select_action_sample_prior_policy]  Prior policy is {' '.join([f'{x:.2f}' for x in prior])}")
        return np.random.choice(len(prior), p=prior)
        
        
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
    
    
    def get_prior_policy(self) -> np.ndarray :
        """
        Get the prior policy.
        """
        prior = []
        for a_idx in range(self.game_board.horizontal_size) :
            if a_idx not in self.actions :
                prior.append(0)
                continue
            prior.append(self.child_priors[a_idx])
        prior = np.array(prior)
        prior = prior / prior.sum()
        return prior
    
    
    def simulate(self, max_sim_steps:int=-1, discount:float=1.) -> float :
        """
        Return value of a game from player X's perspective
        """
    
        ##  Check if game has already been won
        result = self.game_board.result
        if result :
            logger.log(DebugLevel.VERY_HIGH, f"[Node_NeuralMCTS.simulate]  Leaf node found with result {result.name}")
            return result.get_game_score_for_player(BinaryPlayer.X)
                  
        ##  Otherwise return the NN value for this state, which is alway from player X's point of view
        logger.log(DebugLevel.VERY_HIGH, f"[Node_NeuralMCTS.simulate]  Simulation using prior value {self.prior_value:.4f}")
        return self.prior_value



###=======================================###
###   Node_VanillaMCTS class definition   ###
###=======================================###

class Node_VanillaMCTS(Node_Base) :
    
    def __init__(self, game_board:GameBoard, parent:Node_Base=None, params:list=[2.], shallow_copy_board:bool=False, 
                 action:int=-1, label:str=None, noise_lvl:float=0.25, policy_strategy:PolicyStrategy=PolicyStrategy.GREEDY_POSTERIOR_VALUE) :
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

            > noise_lvl, float [0,1], default=0.25
              fraction of noise to use when policy strategy is set to NOISY_POSTERIOR_POLICY
              
            > policy_strategy, PolicyStrategy, default=PolicyStrategy.GREEDY_POSTERIOR_VALUE
              strategy for choosing actions
        """
        
        ##  Call Node_Base initialiser with params=[UCB_c]
        super().__init__(game_board, parent, params, shallow_copy_board, action, label, noise_lvl, policy_strategy)
        
        
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
        
    
    def simulate(self, max_sim_steps:int=-1, discount:float=1.) -> float :
        """
        Simulate a game starting from this node.
        Assumes that both players act according to a uniform-random policy.
        
        Inputs:
        
            > max_sim_steps, int, default=-1
              if positive then determines how many moves to play before declaring a drawn game
              
        Returns:
        
            > float
              the score of the simulation, defined as +1 is X wins, -1 if O wins, 0 for a draw
        """

        ##  Check if game has already been won
        ##  - if so then return score
        ##  - score is -1 if target player has lost, +1 if they've won, and 0 for a draw
        result = self.game_board.result
        if result :
            logger.log(DebugLevel.VERY_HIGH, f"[Node_VanillaMCTS.simulate]  Leaf node found with result {result.name}")
            return result.get_game_score_for_player(BinaryPlayer.X)
                
        ##  Create copy of game board to play simulation
        simulated_game = self.game_board.deep_copy()
        
        ##  Keep playing moves until one of terminating conditions is reached:
        ##  1. game is won by a player
        ##  2. no further moves are possible, game is considered a draw
        ##  3. maximum move limit is reached, game is considered a draw
        turn_idx, compound_discount, result = 0, 1., GameResult.NONE
        trajectory = []
        while not result :
            turn_idx += 1
            compound_discount *= discount
            action = np.random.choice(simulated_game.get_unfilled_columns())
            trajectory.append(f"{simulated_game.to_play.name}:{action}")
            simulated_game.apply_action(action)
            result = simulated_game.result
                  
        ##  Debug trajectory
        logger.log(DebugLevel.VERY_HIGH, f"[Node_VanillaMCTS.simulate]  Simulation ended with result {result.name} with compound_discount={compound_discount:.3f}")
        logger.log(DebugLevel.VERY_HIGH, f"[Node_VanillaMCTS.simulate]  Simulated trajectory was: {' '.join(trajectory)}")
                                
        ##  Return score
        return compound_discount*result.get_game_score_for_player(BinaryPlayer.X)
