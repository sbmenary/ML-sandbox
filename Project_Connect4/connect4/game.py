###
###  connect4.game.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Definition of environment for playing a game of Connect 4.
"""

from __future__ import annotations
from enum       import IntEnum
from colorama   import Fore, Back, Style

import numpy as np

from connect4.utils import DebugLevel



###===================================###
###   BinaryPlayer class definition   ###
###===================================###

class BinaryPlayer(IntEnum):
    """
    An enumeration for the player in a two-player game
    Options are: X=1, 0=-1, NONE=0
    """
    NONE = 0
    X    = 1
    O    = -1
    def label(self) -> str:
        """
        Returns a str representation of the player.
            > X    = 'X'
            > 0    = '0'
            > NONE = '.'
        """
        if self.value == 0  : return '.'
        if self.value == 1  : return f'{Fore.RED}X{Style.RESET_ALL}'
        if self.value == -1 : return f'{Fore.BLUE}O{Style.RESET_ALL}'
        raise NotImplementedError()
    
    @classmethod
    def from_game_result(cls, result:GameResult) -> BinaryPlayer :
        """
        Instantiate game result from a GameResult instance.
        """
        if result == GameResult.X    : return BinaryPlayer.X
        if result == GameResult.O    : return BinaryPlayer.O
        if result == GameResult.DRAW : return BinaryPlayer.NONE
        if result == GameResult.NONE : return BinaryPlayer.NONE
        raise NotImplementedError(f"Could not cast {result} into a BinaryPlayer")

    @classmethod
    def invert(cls, other) -> BinaryPlayer:
        """
        Create BinaryPlayer with inverted value.
        """
        return cls(-other.value)



###================================###
###   GameBoard class definition   ###
###================================###

class GameBoard :
    
    def __init__(self, 
                 horizontal_size: int=7, 
                 vertical_size  : int=6,
                 target_length  : int=4) -> None :
        """
        Class GameBoard
        
        Stores the current state of a game. Allows actions to be played, modifying the internal state. Provides
        simple ASCII visualisation of the board. Internal state stored as numpy array of int objects, using
        BinaryPlayer class to define the IntEnum values.
        
        Inputs:
        
            >  horizontal_size, int, default=7
               horizontal size of the game board
               
            >  vertical_size, int, default=6
               vertical size of the game board
               
            >  target_length, int, default=4
               number of connected pieces required to win the game
        """
        
        ##  Make sure inputs are correctly typed
        if type(horizontal_size) is not int : 
            raise TypeError(f"Expected argument horizontal_size of type int but {type(horizontal_size)} provided")
        if type(vertical_size) is not int : 
            raise TypeError(f"Expected argument vertical_size of type int but {type(vertical_size)} provided")
        if type(target_length) is not int : 
            raise TypeError(f"Expected argument vertical_size of type int but {type(target_length)} provided")
            
        ##  Store game configuration
        self.horizontal_size = horizontal_size
        self.vertical_size   = vertical_size
        self.target_length   = target_length
        
        ##   Initialise game board
        self.board = np.full(shape      = (horizontal_size, vertical_size), 
                             fill_value = BinaryPlayer.NONE.value, 
                             dtype      = np.int8)
        self.to_play         = BinaryPlayer.X
        self.applied_actions = []
        
    
    def __eq__(self, other:GameBoard) -> bool :
        """
        Overload comparison operator comparing two GameBoard objects.
        """
        
        ##  Check games are equally configured
        if self.horizontal_size != other.horizontal_size : return False
        if self.vertical_size   != other.vertical_size   : return False
        if self.target_length   != other.target_length   : return False
        
        ##  Check it's the same player's turn in both games
        if self.to_play != other.to_play : return False
        
        ##  Check game boards are identical
        if (self.board != other.board).any() : return False
        
        ##  If here then games are identical!
        return True
        
        
    def __str__(self) -> str :
        """
        Return a string representation of the game board.
        String is a simple ASCII picture of the game.
        """
        ##  Populate multi-line string with the following steps:
        ##  1. Create empty string to iteratively add lines to
        ##  2. Add upper boundary line
        ##  3. Add graphic for every row in the game board, with the (0,0) located at bottom-left
        ##     - get piece ASCII character representation using label() method for each BinaryPlayer(p) token 
        ##  4. Add middle boundary
        ##  5. Add numerical label for each column
        ##  6. Add lower boundary
        ##  7. Add result label
        ret = ""
        ret += "+-"   +    '-+-'.join(["-" for p in range(self.horizontal_size)]) + "-+"
        ret += "\n| " + ' |\n| '.join([' | '.join([BinaryPlayer(p).label() for p in row]) for row in self.board.T[::-1]]) + " |"
        ret += "\n+-" +    '-+-'.join(["-" for p in range(self.horizontal_size)]) + "-+"
        ret += "\n| " +     '| '.join([f"{p}".ljust(2) for p in range(self.horizontal_size)]) + "|"
        ret += "\n+-" +    '-+-'.join(["-" for p in range(self.horizontal_size)]) + "-+"
        ret += f"\nGame result is: {self.get_result().name}"
        ##  Return complete multi-line str
        return ret
    
    
    def apply_action(self, column_idx: int) -> None :
        """
        Play a new piece at the specified column index. Player is determined by internal state, 
        which keeps track of whose turn it is. If column is full then throw error.
        """
        
        ##  Check that input is correctly typed
        if not np.issubdtype(type(column_idx), int) :
            raise TypeError(f"column_idx of type {type(column_idx)} where int expected")
            
        ##  Check that game has not already finished
        if self.get_result() :
            raise RuntimeError("Cannot play new moves because the game is a terminal state")
            
        ##  Get column from internal numpy array
        column = self.board[column_idx]
        
        ##  Find the smallest unoccupied row index
        row_idx = 0
        while column[row_idx] :
            row_idx += 1
            
        ##  Add piece to the specified (row, column) indices
        column[row_idx] = self.to_play.value
        self.applied_actions.append((column_idx, row_idx, column[row_idx]))
        
        ##  Update whose turn it is to play
        self.to_play = BinaryPlayer.invert(self.to_play)
        
        
    def deep_copy(self) -> GameBoard :
        """
        Create a deep copy of the game board.
        """
        
        ##  Initialise a new GameBoard object with the same configuration
        ##    then perform a deep copy of the internal numpy array, other constants, and return
        new_gameboard                 = GameBoard(self.horizontal_size, self.vertical_size, self.target_length)
        new_gameboard.board           = self.board.copy()
        new_gameboard.to_play         = self.to_play
        new_gameboard.applied_actions = self.applied_actions
        return new_gameboard
        
        
    @classmethod
    def from_gameboard(cls, gameboard) -> GameBoard :
        """
        Create a new GameBoard object as a deep copy of one provided.
        """
        return gameboard.deep_copy()
            
    
    def get_available_actions(self) -> list[int] :
        """
        Get list of available actions. Action corresponds to a column index. Action is considered 
        available if the game has not ended and column is not already full.
        """
        
        ##  Check whether game has ended
        if self.get_result() :
            return []
        
        ##  Otherwise return list of all unfilled column indices
        return self.get_unfilled_columns()
            
    
    def get_unfilled_columns(self) -> list[int] :
        """
        Get list of unfilled columns
        """
        return [a for a in range(self.horizontal_size) if self.board[a,-1] == 0]
    
    
    def get_result(self) -> GameResult :
        """
        Check whether the game board has reached a terminal state.
        """
        
        ##  Check for vertical win condition
        for column in self.board :
            last_piece, counter, row_idx = 999, 1, 0
            while last_piece and row_idx < self.vertical_size :
                piece = column[row_idx]
                if piece == last_piece :
                    counter += 1
                else :
                    counter = 1
                if counter == self.target_length :
                    return GameResult.from_piece_value(BinaryPlayer(piece))
                row_idx += 1
                last_piece = piece
                
        ##  Check for horizontal win condition
        for row in self.board.T :
            last_piece, counter = 999, 1
            for col_idx in range(self.horizontal_size) :
                piece = row[col_idx]
                if piece == last_piece :
                    counter += 1
                else :
                    counter = 1
                if piece != 0 and counter == self.target_length :
                    return GameResult.from_piece_value(piece)
                last_piece = piece
                
        ##  Check for diagonal win condition
        for col_idx in range(self.horizontal_size) :
            for row_idx in range(self.vertical_size) :
                piece = self.board[col_idx, row_idx]
                if not piece : continue
                for col_dir, row_dir in [[1,1], [1,-1], [-1,1], [-1,-1]] :
                    is_winning_sequence = True
                    for seq_idx in range(self.target_length) :
                        check_col_idx, check_row_idx = col_idx + col_dir*seq_idx, row_idx + row_dir*seq_idx
                        if (check_col_idx < 0 or check_col_idx >= self.horizontal_size or 
                            check_row_idx < 0 or check_row_idx >= self.vertical_size or 
                            self.board[check_col_idx, check_row_idx] != piece) :
                            is_winning_sequence = False
                            break
                    if is_winning_sequence :
                        return GameResult.from_piece_value(piece)
                    
        ##  Check for draw
        if len(self.get_unfilled_columns()) == 0 :
            return GameResult.DRAW
        
        ##  If here then game has not finished
        return GameResult.NONE
    
    
    def undo_action(self) -> None :
        """
        Undo the most recent action stored in the internal record. 
        A full game record is maintained, allowing many moves to be undone.
        """
        
        ##  Check that an action exists
        if len(self.applied_actions) == 0 :
            raise RuntimeError(f"No actions to undo.")
            
        ##  Remove and return the last item from the record of applied actions
        column, row, _ = self.applied_actions.pop()
        
        ##  Return the specific index to its 0 state, indicating that no piece is present
        self.board[column, row] = 0
    


###=================================###
###   GameResult class definition   ###
###=================================###

class GameResult(IntEnum):
    """
    An enumeration for the result of a two-player game
    NONE=0 enforced to allow "is GameResult" statements to return True only if a game has ended
    """
    NONE = 0
    DRAW = 1
    X    = 2
    O    = 3
        

    @classmethod
    def from_piece_value(cls, value:int, none_player_means_draw:bool=True) -> GameResult :
        """
        Instantiate game result from a BinaryPlayer value. 
        
        Inputs:
        
            >  value, int
               value to be converted to BinaryPlayer instance, and then to GameResult
               
            >  none_player_means_draw, bool, default=True
               if True then interpret BinaryPlayer.NONE as GameResult.DRAW, otherwise GameResult.NONE
        """
        return cls.from_player(BinaryPlayer(value), none_player_means_draw=none_player_means_draw)
    

    @classmethod
    def from_player(cls, player:BinaryPlayer, none_player_means_draw:bool=True) -> GameResult :
        """
        Instantiate game result from a BinaryPlayer instance.
        
        Inputs:
        
            >  player, BinaryPlayer
               instance of BinaryPlayer to be converted to GameResult
               
            >  none_player_means_draw, bool, default=True
               if True then interpret BinaryPlayer.NONE as GameResult.DRAW, otherwise GameResult.NONE
        """
        if player == BinaryPlayer.X    : return GameResult.X
        if player == BinaryPlayer.O    : return GameResult.O
        if player == BinaryPlayer.NONE : 
            if none_player_means_draw : return GameResult.DRAW
            return GameResult.NONE
        raise NotImplementedError(f"Could not cast {player} into a GameResult")
    

    def get_game_score_for_player(self, player:BinaryPlayer) -> float :
        """
        Return the game score for the given player.
        Score is 0 for a DRAW or NONE, +1 if the GameResult matches the BinaryPlayer, -1 otherwise
        
        Inputs:
        
            >  player, BinaryPlayer
               enum of the player to whom the score applies
        """
        
        ##  Require player to be resolved
        if player not in [BinaryPlayer.X, BinaryPlayer.O] :
            raise NotImplementedError(f"Cannot resolve score for player {player.name}")
        
        ##  Resolve NONE result
        if self == GameResult.NONE :
            return 0.
        
        ##  Resolve DRAW result
        if self == GameResult.DRAW :
            return 0.
        
        ##  Resolve X WIN result for X PLAYER
        if self == GameResult.X and player == BinaryPlayer.X :
            return 1.
        
        ##  Resolve O WIN result for O PLAYER
        if self == GameResult.O and player == BinaryPlayer.O :
            return 1.
        
        ##  If here then the specifid player must have lost
        return -1.
        