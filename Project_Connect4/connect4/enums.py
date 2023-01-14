###
###  connect4.enums.py
###  author: S. Menary [sbmenary@gmail.com]
###
"""
Various enums to make life easier.
"""

from __future__ import annotations    ## Allows type-hint to reference same class being defined, __future__ must be imported first
from enum import IntEnum



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
        Returns a single character representation of the player.
            > X    = 'X'
            > 0    = '0'
            > NONE = '.'
        """
        if self.value == 0  : return '.'
        if self.value == 1  : return 'X'
        if self.value == -1 : return '0'
        raise NotImplementedError()

    @classmethod
    def invert(cls, other) -> BinaryPlayer:
        """
        Create BinaryPlayer with inverted value.
        """
        return cls(-other.value)



###=================================###
###   DebugLevel class definition   ###
###=================================###

class DebugLevel(IntEnum):
    """
    An enumeration for the verbosity level of debug statements
    Options are: MUTE=0, LOW=1, MEDIUM=2, HIGH=3, ALL=4
    Use of IntEnum allows labelled int operations like "if lvl >= DebugLevel.MEDIUM :"
    """
    MUTE   = 0
    LOW    = 1
    MEDIUM = 2
    HIGH   = 3
    ALL    = 4
    
    def message(self, min_lvl:DebugLevel, message:str) -> bool :
        """
        Print message only if self >= min_lvl.
        Returns bool indicating whether the message was printed.
        """

        ##  If debug_lvl >= min_lvl then print message and return True
        if self >= min_lvl :
            print(message)
            return True

        ##  Otherwise no message printed and return False
        return False
    


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

        