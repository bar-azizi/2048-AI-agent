import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in
                  legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if
                        scores[index] == best_score]

        chosen_index = np.random.choice(
            best_indices)  # Pick randomly among the bes
        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        # empty tiles of the previous state
        prev_zero_count = np.sum(current_game_state.board == 0)
        # generating successor (the current state)
        successor_game_state = current_game_state.generate_successor(
            action=action)
        # board state of successor
        board = successor_game_state.board
        # The tile with the nax value in the successor board
        max_tile = successor_game_state.max_tile
        # Number of empty tiles in the successor board
        zero_count = np.sum(board == 0)

        # Count merges done with the action
        merges = zero_count - prev_zero_count
        two_count = np.sum(board == 2)
        four_count = np.sum(board == 4)

        # score bonuses for max tile in first line and for right up edge
        score = successor_game_state.score
        if max_tile in board[0, :]:
            score += 20
        if board[0, 0] == max_tile:
            score += 20
        # checking number of tiles with max_tile value
        max_tile_count = np.sum(board[0, :] == max_tile)
        # checking number of tiles with max_tile//2 value
        big_tile_count = np.sum(board[0, :] == max_tile / 2)

        # return the value of the state based on what we calculated
        return score * 2 + max_tile_count + big_tile_count + merges * 10 - (16
                                                                            - zero_count) * 2


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        # if we do not need to look on next moves:
        if self.depth == 0:
            return Action.STOP
        # returning the value of the minmax function
        return self.minimax(game_state, self.depth*2, 0)[1]

    def minimax(self, game_state, depth, player):
        # set actions according to player (agent or opponent)
        actions = game_state.get_legal_actions(player)

        # if we reached to end of game or that we reached to the end of
        # given depth, return the value of the state
        if depth == 0 or not actions:
            return self.evaluation_function(game_state), Action.STOP

        # player is the maximum player
        if player == 0:
            # best score initializes to -infinity, so the first score will be
            # change the best score
            best_score = float('-inf')
            best_action = None
            # go through all possible actions
            for action in actions:
                successor = game_state.generate_successor(player, action)
                # take the minmax value of the next turn (other player)
                score, next_action = self.minimax(successor, depth-1, 1)
                # check if the score we got in minmax of successor is higher
                # than the highest score so far
                if score > best_score:
                    # update the highest score
                    best_score = score
                    best_action = action
            # return the highest score we got and the action that led to the
            # score
            return best_score, best_action

        # player is the minimum player (opponent)
        if player == 1:
            # best score initializes to infinity, so the first score will be
            # change the best score
            best_score = float('inf')
            best_action = None
            # go through all possible actions
            for action in actions:
                successor = game_state.generate_successor(player, action)
                # take the minmax value of the next turn (other player)
                score, next_action = self.minimax(successor, depth-1, 0)
                # check if the score we got in minmax of successor is lower
                # than the lowest score so far
                if score < best_score:
                    # update the lowest score
                    best_score = score
                    best_action = action
            # return the lowest score we got and the action that led to the
            # score
            return best_score, best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        # if we do not need to look on next moves:
        if self.depth == 0:
            return Action.STOP
        # returning the value of the alphabeta function
        return self.alphabeta(game_state, self.depth*2, 0, float('inf'), 0)[1]

    def alphabeta(self, game_state, depth, alpha, beta, player):
        # set actions according to player (agent or opponent)
        actions = game_state.get_legal_actions(player)

        # if we reached to end of game or that we reached to the end of
        # given depth, return the value of the state
        if depth == 0 or not actions:
            return self.evaluation_function(game_state), Action.STOP

        # if player is maximum player
        if player == 0:
            # best score initializes to -infinity, so the first score will be
            # change the best score
            best_score = float('-inf')
            best_action = None
            # go through all possible actions
            for action in actions:
                successor = game_state.generate_successor(player, action)
                # take the alphabeta value of the next turn (other player)
                score, next_action = self.alphabeta(successor, depth-1, alpha, beta, 1)
                # check if the score we got in minmax of successor is higher
                # than the highset score so far
                if score > best_score:
                    # update the highest score
                    best_score = score
                    best_action = action
                # check if best score is larger than alpha, than update alpha
                # if needed
                alpha = max(best_score, alpha)
                # no need to search more down the tree if beta <= alpha
                if beta <= alpha:
                    break
            # return the highest score we got and the action that led to the
            # score
            return best_score, best_action

        # if player is minimum player
        if player == 1:
            # best score initializes to infinity, so the first score will be
            # change the best score
            best_score = float('inf')
            best_action = None
            # go through all possible actions
            for action in actions:
                successor = game_state.generate_successor(player, action)
                # take the alphabeta value of the next turn (other player)
                score, next_action = self.alphabeta(successor, depth-1, alpha, beta, 0)
                # check if the score we got in minmax of successor is lower
                # than the lowest score so far
                if score < best_score:
                    # update the lowest score
                    best_score = score
                    best_action = action
                # check if best score is smaller than beta, than update beta
                # if needed
                beta = min(best_score, beta)
                # no need to search more down the tree if beta <= alpha
                if beta <= alpha:
                    break
            # return the lowest score we got and the action that led to the
            # score
            return best_score, best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        # if we do not need to look on next moves:
        if self.depth == 0:
            return Action.STOP
        # returning the value of the expectimax function
        return self.expectimax(game_state, self.depth, 0, float('inf'), 0)[1]

    def expectimax(self, game_state, depth, alpha, beta, player):
        # set actions according to player (agent or opponent)
        actions = game_state.get_legal_actions(player)

        # if we reached to end of game or that we reached to the end of
        # given depth, return the value of the state
        if depth == 0 or not actions:
            return self.evaluation_function(game_state), Action.STOP

        # if player is maximum player
        if player == 0:
            # best score initializes to -infinity, so the first score will be
            # change the best score
            best_score = float('-inf')
            best_action = None
            # go through all possible actions
            for action in actions:
                successor = game_state.generate_successor(player, action)
                # take the expectimax value of the next turn (other player)
                score, next_action = self.expectimax(successor, depth, alpha, beta, 1)
                # check if the score we got in minmax of successor is lower
                # than the lowest score so far
                if score > best_score:
                    best_score = score
                    best_action = action
                # check if best score is larger than alpha, than update alpha
                # if needed
                alpha = max(best_score, alpha)
                # no need to search more down the tree if beta <= alpha
                if beta <= alpha:
                    break
            # return the highest score we got and the action that led to the
            # score
            return best_score, best_action

        # if player is the opponent (chooses a move randomly)
        if player == 1:
            # initialize the expected (average in our case) value, and the
            # best score
            expected = 0
            best_score = 0
            num_actions = len(actions)
            # go through all possible actions
            for action in actions:
                successor = game_state.generate_successor(player, action)
                # take the expectimax value of the next turn (other player)
                score, next_action = self.expectimax(successor, depth - 1, alpha, beta, 0)
                # add scores to expected
                expected += score
                # update bes score if needed
                if score > best_score:
                    best_score = score
                # update beta if needed but with no use in this turn
                beta = min(best_score, beta)
            # calculate acerage
            expected = expected / num_actions
            # return the expected value, but no best action
            return expected, None  # ,best_action


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: In this function, we tried to create a situation in which the
    max tile sticks to one of the corners, and the larger tiles on the board
    that are close to the max tile as much as possible in a decreasing order.
    In this situation, it would be easier to merge a few tiles in a row.
In addition, the bonus for the max tile on one of the edges depends on the
value of the max tile, so it will be proportional to the score of the game
(every merge of a large max tile such as 1024 or 2048 makes the score a lot
higher)
Furthermore, we gave two types of penalties – a penalty for every tile on the
board (multiplied by 2), and a penalty for the difference between two nearby
tiles on the board(smoothness). The smoothness is the lowest when two nearby
tiles are closest to each other (like 2048 and 1024 vs 2048 and 512)  which
gives a boost to the states with monotonicity in a row, especially on the row
with the max tile in it.

    """
    "*** YOUR CODE HERE ***"
    # initialize board from current state and max_tile as the max tile on board
    board = current_game_state.board
    max_tile = current_game_state.max_tile
    # count the number of empty tiles in the board
    zero_count = np.sum(board == 0)

    # current score of the game
    score = current_game_state.score

    # add points for max_tile in first row
    if max_tile in board[0, :]:
        score += 20
        # add points if the max_tile in right edge or left edge of
        # the first row
        if board[0, 0] == max_tile or board[0, 3] == max_tile:
            score += 4*max_tile
        # count the times max_tile is showing in first line
        max_tile_count = np.sum(board[0, :] == max_tile)
        # count the times second-biggest tile is showing in first line
        big_tile_count = np.sum(board[0, :] == max_tile / 2)

    # add points for max_tile in last row
    elif max_tile in board[3, :]:
        score += 20
        # add points for max_tile on edges
        if board[3, 0] == max_tile or board[3, 3] == max_tile:
            score += 4*max_tile

        # calculate number of max tile on board and number of second max tile
        # on board
        max_tile_count = np.sum(board[3, :] == max_tile)
        big_tile_count = np.sum(board[3, :] == max_tile / 2)
    else:
        max_tile_count = np.sum(board[0, :] == max_tile)
        big_tile_count = np.sum(board[0, :] == max_tile / 2)

    # punish that gets bigger then the difference between two nearby tiles is
    # bigger
    smoothness = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1] - 1):
            smoothness += abs(board[i, j] - board[i, j + 1])
    for j in range(board.shape[1]):
        for i in range(board.shape[0] - 1):
            smoothness += abs(board[i, j] - board[i + 1, j])

    # for every merge in the state, adds the tile that cam be merged to the
    # score
    merges = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1] - 1):
            if board[i, j] == board[i, j + 1]:
                merges += board[i, j]
    for j in range(board.shape[1]):
        for i in range(board.shape[0] - 1):
            if board[i, j] == board[i + 1, j]:
                merges += board[i, j]

    # calculate the return value
    return score * 2 + max_tile_count + big_tile_count + max_tile - (
                16 - zero_count) * 2 - smoothness * 2 + merges


# Abbreviation
better = better_evaluation_function
