"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import logging

logger = logging.getLogger(__name__)


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def moves_number(game, player):
    """util function that gives moves of player and opponent

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).
    :return: moves of player, opponent
    """
    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return player_moves, opponent_moves


def check_util(game, player, diff):
    """ utility function to check the utility

   Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    :return: return diff if util == 0. else util
    """
    util = game.utility(player)
    return diff if util == 0. else util


def h_move_improve_weighted(game, player):
    """Heuristic uses weighted improved score

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    player_moves, opponent_moves = moves_number(game, player)
    player_weight = 1.9
    opponent_weight = 0.7
    diff = float(player_moves * player_weight - opponent_moves * opponent_weight)
    return check_util(game, player, diff)


def h_weighted_game_height_move_count(game, player):
    """Heuristic to improve by using game move and height ratio as compared to the
    opponent's moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    player_moves, opponent_moves = moves_number(game, player)
    diff = float(player_moves - opponent_moves * (game.move_count / game.height))
    return check_util(game, player, diff)


def h_game_size_improve(game, player):
    """Game size, using the middle ground of the game to decide the improvement.
    else it is just simple improved_score.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    player_moves, opponent_moves = moves_number(game, player)
    game_total = game.width * game.height
    game_move_count = game.move_count
    if game_total * 0.3 <= game_move_count <= game_total * 0.7:
        diff = float(player_moves - opponent_moves * (game.move_count / game.height))
    else:
        diff = float(player_moves - opponent_moves)
    return check_util(game, player, diff)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    logger.debug("h_move_improve_weighted: ", h_move_improve_weighted(game, player))
    logger.debug("h_game_size_improve: ", h_game_size_improve(game, player))
    logger.debug("h_weighted_game_height_move_count: ",
                 h_weighted_game_height_move_count(game, player))

    return h_weighted_game_height_move_count(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.POSITIVE_INF = float("inf")
        self.NEGATIVE_INF = float("-inf")
        self.__init_method__()

    def __init_method__(self):
        """
        Helper method: Initialize the alias to the proper searcg method.
        Sets the search method alias and also sets the method dictionary
            that maps string names to aliases.
        :return: set and return the alias to the proper method.
        """
        self.method_map = {
            'minimax': self.minimax,
            'alphabeta': self.alphabeta
        }
        self.method_alias = self.method_map[self.method]
        return self.method_alias

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if len(legal_moves) <= 0:
            return (-1, -1)

        best = legal_moves[0]  # set to the first one or whatever you want

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                depth = 1
                while True:  # let the time run out, better if we use "score"
                    score, best = self.method_alias(game, depth, maximizing_player=True)
                    depth += 1
            else:
                depth = self.search_depth
                score, best = self.method_alias(game, depth, maximizing_player=True)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            logger.debug("I, the agent, ran out of time. Returning whatever I can")
            return best

        # Return the best move from the last completed search iteration
        return best

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD + 2:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), (-1, -1)

        branches = []
        legal_moves_current_game = game.get_legal_moves()

        if depth == 1:  # specical case, just get the scores of all legal moves
            for move in legal_moves_current_game:
                next_state = game.forecast_move(move)
                branches.append((self.score(next_state, self), move))
        else:
            for move in legal_moves_current_game:
                next_state = game.forecast_move(move)
                # need the score form minimax for this move, combined as this tuple
                branches.append(
                    (self.minimax(next_state, depth - 1, not maximizing_player)[0], move))

        # set the eval function as alias for min or max depending on
        # if the player is maximizing or not
        evaluation_func = max if maximizing_player else min
        if len(branches) <= 0:
            return (-1, -1)
        return evaluation_func(branches)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD + 2:
            raise Timeout()

        legal_moves_current_game = game.get_legal_moves()
        if depth == 0 or len(legal_moves_current_game) <= 0:  # special case
            return self.score(game, self), (-1, -1)

        best_move = legal_moves_current_game[0]  # set to the first one or whatever you want
        # default values if extremes are set in the __init__ itself
        best_score = self.NEGATIVE_INF if maximizing_player else self.POSITIVE_INF

        # explicit is better than implicit
        new_alpha = alpha
        new_beta = beta

        for move in legal_moves_current_game:
            next_state = game.forecast_move(move)
            # depth first, to find the score
            score, move_dont_care = self.alphabeta(next_state, depth - 1, new_alpha, new_beta,
                                                   not maximizing_player)
            if (maximizing_player and score > best_score) or (
                        not maximizing_player and score < best_score):
                best_move = move
                best_score = score
                # no need to traverse further if maximizing and the score is smaller than
                # or equal to what we already have, and if non-maximizing turn then
                # greater than or equal to what we have. So break out of the branch
                if (maximizing_player and beta <= best_score) or (
                            not maximizing_player and alpha >= best_score):
                    logger.debug("Breaking out because this branch is not required")
                    break
                new_alpha = best_score if maximizing_player else alpha
                new_beta = beta if maximizing_player else best_score
        return best_score, best_move
