# T. Sorg
# Advanced AI Combinatorial Games
import time
from numpy import random


class State(object):
    def __init__(self):
        # Move
        self.move_number = 1
        # Which player to move now
        self.current_player = 'W'  # alternates between 'W' and 'B'
        # Board
        self.board = []
        row0 = 'kqbnr'
        row1 = 'ppppp'
        row2 = '.....'
        row3 = '.....'
        row4 = 'PPPPP'
        row5 = 'RNBQK'
        strings = [row0, row1, row2, row3, row4, row5]
        for row in strings:
            self.board.append([letter for letter in row])
        self.piece_values = {'k': 200, 'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1,
                             'K': 200, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1}
        self.column_or_file_conversion = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        self.inverse_column_or_file_conversion = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

        # For tracking current pieces on board
        self.white_pieces = []
        self.black_pieces = []
        # Where the magic happens
        self.keepTrackOfMaterial()

        # For undo functionality
        self.previous_states = []

        # Iterative deepening variables
        self.time_cache = 0
        self.time_counter = 0  # Update the cache after a certain number of increments
        self.time_limit = 0  # For the current move

    # Track the pieces on the board for each player and score them accordingly
    def keepTrackOfMaterial(self):
        self.black_pieces = []
        self.white_pieces = []
        for row in self.board:
            for piece in row:
                if piece.isupper():  # is White
                    self.white_pieces.append(piece)
                elif piece.islower():  # is Black
                    self.black_pieces.append(piece)

    def reset(self):
        self.__init__()

    def experimental_eval_scores(self, vals):
        self.piece_values['q'] = vals[0]
        self.piece_values['r'] = vals[1]
        self.piece_values['b'] = vals[2]
        self.piece_values['n'] = vals[3]
        self.piece_values['p'] = vals[4]
        self.piece_values['Q'] = vals[0]
        self.piece_values['R'] = vals[1]
        self.piece_values['B'] = vals[2]
        self.piece_values['N'] = vals[3]
        self.piece_values['P'] = vals[4]

    def getBoard(self):
        stream = str(self.move_number) + ' '
        stream += self.current_player + '\n'
        for i in range(6):
            stream += ''.join(self.board[i]) + '\n'
        return stream

    def setBoard(self, stream):
        fields = stream.split('\n')
        top_line = fields[0].split(' ')
        self.move_number = int(top_line[0])
        self.current_player = top_line[1]
        board = fields[1:-1]
        self.board = [[letter for letter in row] for row in board]
        self.keepTrackOfMaterial()

    def winner(self):
        less_than_41_moves = self.move_number < 41
        black_has_king = False
        white_has_king = False
        for row in self.board:
            if 'k' in row:
                black_has_king = True
            if 'K' in row:
                white_has_king = True
        if black_has_king and white_has_king and less_than_41_moves:
            return '?'
        elif black_has_king and white_has_king and not less_than_41_moves:
            return '='
        elif not black_has_king and white_has_king:
            return 'W'
        elif black_has_king and not white_has_king:
            return 'B'
        elif not black_has_king and not white_has_king:  # cheesy way to avoid segfault
            return '='

    def isPiece(self, strPiece):
        return strPiece in 'KQBNRPkqbnrp'

    def isEnemy(self, strPiece):
        is_black = strPiece in 'kqbnrp'
        is_white = strPiece in 'KQBNRP'
        is_empty = strPiece == '.'
        if self.current_player == 'W':
            if is_black:
                return True
            elif is_white or is_empty:
                return False
        elif self.current_player == 'B':
            if is_white:
                return True
            elif is_black or is_empty:
                return False

    def isOwn(self, strPiece):
        is_black = strPiece in 'kqbnrp'
        is_white = strPiece in 'KQBNRP'
        is_empty = strPiece == '.'
        if self.current_player == 'W':
            if is_white:
                return True
            elif is_black or is_empty:
                return False
        elif self.current_player == 'B':
            if is_black:
                return True
            elif is_white or is_empty:
                return False

    def isNothing(self, strPiece):
        if strPiece == '.':
            return True
        else:
            return False

    def isValid(self, x, y):
        if x < 0:
            return False
        elif x > 4:
            return False
        if y < 0:
            return False
        elif y > 5:
            return False
        return True

    def moreComplicatedEval(self):  # Slower and I'm not yet satisfied with the decisions made using this eval.
        player_perspective = 1 if self.current_player == 'W' else -1
        material_scalar = 1.0
        mobility_scalar = 0.01
        K, Q, R, B, N, P = self.white_pieces.count("K"), self.white_pieces.count("Q"), self.white_pieces.count(
            "R"), self.white_pieces.count("B"), self.white_pieces.count("N"), self.white_pieces.count("P")
        k, q, r, b, n, p = self.black_pieces.count("k"), self.black_pieces.count("q"), self.black_pieces.count(
            "r"), self.black_pieces.count("b"), self.black_pieces.count("n"), self.black_pieces.count("p")
        material = 200 * (K - k) + 9 * (Q - q) + 5 * (R - r) + 3 * (B - b + N - n) + (P - p)
        white_mobility, black_mobility = self.calculateMobility()
        # TODO: pawn structure calculations
        # TODO: piece placement (each location has its own bonus or penalty

        return (material_scalar * material + mobility_scalar * (white_mobility - black_mobility)) * player_perspective

    def eval(self):  # TODO: cache the eval scores and update when moving/undo-ing to make it faster(?)
        # Evaluate the board in the perspective of current player
        player = 1 if self.current_player == 'W' else -1
        white_points = sum([self.piece_values[piece] for piece in self.white_pieces])
        black_points = sum([self.piece_values[piece] for piece in self.black_pieces])
        return (white_points - black_points) * player  # faster than if /else... player == black ==> blackpts-whitepts

    def moveScan(self, x0, y0, dx, dy, stop_short=False, capture=True):
        # Get starting coordinates of piece to move
        x = x0
        y = y0
        moves = []
        # Repeat until stop_short is True
        while True:
            # Search in direction <dx,dy>
            x = x + dx
            y = y + dy
            # Check if valid destination square
            if self.isValid(x, y) is False:  # x or y is not in bounds:
                break
            # Get piece at destination (could be empty '.' or a black or white piece.)
            p = self.board[y][x]
            # If there is a piece at the destination
            if self.isPiece(p):
                if self.isOwn(p):
                    break
                # Not the same color. Are we trying to capture? 
                if capture is False:
                    break
                # Capture is True and p is opposite color
                # Clearly we can't search further.
                stop_short = True
            elif capture == 'only':
                break
            # Add valid move to list of moves
            moves.append(self.createMoveString(x0, y0, x, y))
            # Body repeated until...
            if stop_short is True:
                break
        return moves

    def symmScan(self, x0, y0, dx, dy, stop_short=False, capture=True):
        moves = []
        for i in range(4):
            moves.extend(self.moveScan(x0, y0, dx, dy, stop_short, capture))
            dy, dx = -dx, dy  # exchange and negate dy
        return moves

    def moveList(self, x, y):
        p = self.board[y][x]
        moves = []
        if p in 'KQkq':  # King or Queen
            stop_short = p in 'Kk'
            moves.extend(self.symmScan(x, y, 0, -1, stop_short))
            moves.extend(self.symmScan(x, y, 1, -1, stop_short))
        elif p in 'RBrb':  # Rook or Bishop
            stop_short = p in 'Bb'
            capture = p in 'Rr'
            moves.extend(self.symmScan(x, y, 0, -1, stop_short, capture))
            if p in 'Bb':
                moves.extend(self.symmScan(x, y, 1, -1))  # stop_short is False and capture is True by default
        elif p in 'Nn':  # Knight
            moves.extend(self.symmScan(x, y, 1, -2, stop_short=True))  # Capture is True by default
            moves.extend(self.symmScan(x, y, -1, -2, stop_short=True))  # Capture is True by default
        elif p in 'Pp':  # Pawn
            direction = -1  # I have a board that is reversed compared to the tutorial.
            if p == 'p':
                direction = 1
            moves.extend(self.moveScan(x, y, -1, direction, stop_short=True, capture='only'))
            moves.extend(self.moveScan(x, y, 1, direction, stop_short=True, capture='only'))
            moves.extend(self.moveScan(x, y, 0, direction, stop_short=True, capture=False))
        return moves

    def allValidMoves(self):
        moves = []
        if self.winner() != '?':
            return []  # no more valid moves because the game ended.
        for y in range(6):
            for x in range(5):
                if self.isPiece(self.board[y][x]) and self.isOwn(self.board[y][x]):
                    moves.extend(self.moveList(x, y))
        return moves

    def shuffledMoves(self):
        moves = self.allValidMoves()
        random.shuffle(moves)
        return moves

    def letterToNumber(self, letter):
        return self.column_or_file_conversion[letter]

    def numberToLetter(self, x):
        return self.inverse_column_or_file_conversion[x]

    def createMoveString(self, x0, y0, x, y):
        from_string = self.numberToLetter(int(x0)) + str(6 - y0)
        to_string = self.numberToLetter(int(x)) + str(6 - y)
        move_string = from_string + '-' + to_string + '\n'
        return move_string

    # Used by conversion function in move function.
    def decodeMoveString(self, move_string):
        from_string = move_string[:2]
        from_col = self.column_or_file_conversion[from_string[0]]
        from_row = 6 - int(from_string[1])
        from_square = Square(from_col, from_row)
        to_string = move_string[3:]
        to_col = self.column_or_file_conversion[to_string[0]]
        to_row = 6 - int(to_string[1])
        to_square = Square(to_col, to_row)
        return Move(from_square, to_square)

    # Used in move function.
    def convertMoveForm(self, move_string):
        x_coords = move_string[0] in 'abcde' and move_string[3] in 'abcde'
        y_coords = move_string[1] in '123456' and move_string[4] in '123456'
        dash = move_string[2] == '-'
        if x_coords and y_coords and dash:  # valid move
            return self.decodeMoveString(move_string)
        else:
            raise Exception('move string was bad')

    def move(self, move_arg):
        # Take a move argument and apply it to the board. ASSUMES LEGAL MOVE
        if len(move_arg) == 0:
            return
        move = self.convertMoveForm(move_arg)
        from_square = move.from_square
        to_square = move.to_square

        # Undo functionality: Save current board state
        self.previous_states.append([[piece for piece in row] for row in self.board])

        # Get the piece you're moving
        moving_piece = self.board[from_square.y][from_square.x]
        # Get the square's piece you're moving to
        dest = self.board[to_square.y][to_square.x]
        # Make the from square vacant
        self.board[from_square.y][from_square.x] = '.'
        # Overwrite destination square with moving piece
        self.board[to_square.y][to_square.x] = moving_piece

        # Keep track of which piece was attacked and removed
        if dest.islower():  # dest piece is black
            self.black_pieces.remove(dest)
        elif dest.isupper():  # dest piece is black
            self.white_pieces.remove(dest)

        # Pawn promotion conditions
        white_pawn = moving_piece == 'P'
        black_pawn = moving_piece == 'p'
        to_white_backrow = to_square.y == 5
        to_black_backrow = to_square.y == 0

        # Promote white pawn
        if white_pawn and to_black_backrow:
            self.board[to_square.y][to_square.x] = 'Q'
            self.white_pieces.remove('P')
            self.white_pieces.append('Q')
        # Promote black pawn
        elif black_pawn and to_white_backrow:
            self.board[to_square.y][to_square.x] = 'q'
            self.black_pieces.remove('p')
            self.black_pieces.append('q')

        # End White's Turn
        if self.current_player == 'W':
            # Now it's black's turn
            self.current_player = 'B'

        # End Black's Turn
        elif self.current_player == 'B':
            # Now it is white's turn
            self.current_player = 'W'
            # Start next move number
            self.move_number += 1

    def undo(self):
        # TODO: Improving undo: You could save two pieces and their positions to represent a move.
        if len(self.previous_states) > 0:
            # Restore previous state
            self.board = self.previous_states.pop()
            self.move_number = self.move_number - 1 if self.current_player is 'W' else self.move_number
            self.current_player = 'B' if self.current_player is 'W' else 'W'
            # Recalculate
            self.keepTrackOfMaterial()

    def evaluatedMoves(self):
        moves = self.shuffledMoves()
        if len(moves) < 1:
            return []
        eval_moves = []
        for move in moves:
            self.move(move)
            score = self.eval()
            self.undo()
            eval_moves.append([score, move])

        grouped_moves = []
        while len(eval_moves) > 0:
            for i in eval_moves:
                if i[0] == min(eval_moves)[0]:
                    grouped_moves.append(eval_moves.pop(eval_moves.index(i)))

        return [move[1] for move in grouped_moves]  # List the moves without their scores

    def randomMove(self):
        moves = self.allValidMoves()
        if len(moves) > 0:
            move = random.choice(moves)
            return move
        else:
            return ''

    def greedy(self):
        return self.evaluatedMoves()[0]  # First move in loosely ordered (by increasing score) listj

    def negamax(self, intDepth):
        # ###############################################################
        # Check the time.
        self.time_counter += 1
        if self.time_counter > 1000:
            self.time_counter = 0
            self.time_cache = int(time.time() * 1000)
        if self.time_cache > self.time_limit:
            return 0
        # ###############################################################
        # Start the search
        if (intDepth == 0) or (self.winner() != '?'):
            return self.eval()

        score = float('-inf')
        moves = self.evaluatedMoves()
        for move in moves:
            self.move(move)
            score = max(score, -self.negamax(intDepth - 1))
            self.undo()
        return score

    def moveNegamax(self, intDepth, intDuration):
        # ###############################################################
        # Start the clock
        self.time_cache = int(time.time() * 1000)
        self.time_limit = int(self.time_cache + intDuration)
        self.time_counter = 0
        # ###############################################################
        # Start the search
        best = ''
        moves = self.evaluatedMoves()
        for depth in range(1, intDepth + 1):
            score = float('-inf')
            candidate = ''
            for move in moves:
                self.move(move)
                temp = -self.negamax(depth - 1)
                self.undo()
                if temp > score:
                    candidate = move
                    score = temp
            if self.time_cache > self.time_limit:
                break
            best = candidate
        return best

    def alphabeta(self, depth, alpha, beta):
        # ###############################################################
        # Check the time.
        self.time_counter += 1
        if self.time_counter > 1000:
            self.time_counter = 0
            self.time_cache = int(time.time() * 1000)
        if self.time_cache > self.time_limit:
            return 0
        # ###############################################################
        if (depth == 0) or (self.winner() != '?'):
            return self.eval()
        score = float('-inf')
        for move in self.evaluatedMoves():
            self.move(move)
            score = max(score, -self.alphabeta(depth - 1, -beta, -alpha))
            self.undo()
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return score

    def moveAlphabeta(self, intDepth, intDuration):
        # ###############################################################
        # Start the clock
        self.time_cache = int(time.time() * 1000)
        self.time_limit = int(self.time_cache + intDuration)
        self.time_counter = 0
        # ###############################################################
        # Search for best move
        best = ''
        moves = self.evaluatedMoves()
        # Search for the best move starting at depth = 1
        for depth in range(1, intDepth + 1):
            # Reset upper and lower bounds
            alpha = float('-inf')
            beta = float('inf')
            # Reset candidate move
            candidate = ''
            # Search for best candidate at current depth
            for move in moves:
                self.move(move)
                temp = -self.alphabeta(depth - 1, -beta, -alpha)
                self.undo()
                if temp > alpha:
                    candidate = move
                    alpha = temp
            # If we ran out of time at this search depth, don't save the current candidate.
            if self.time_cache > self.time_limit:
                break
            # We still have enough time so save the candidate as the best candidate found so far.
            best = candidate

        # Show current player, move number, depth of search, and time elapsed during search.
        return best

    def print_stats_for_move(self, depth, start):
        print 'color=%s,move=%d,depth=%d,time elapsed=%d' % (self.current_player, self.move_number, depth,
                                                             self.time_cache - start)

    def tournamentMoveAlphabeta(self, intDuration):
        # ###############################################################
        # Start the clock
        self.time_cache = int(time.time() * 1000)
        start = self.time_cache
        self.time_limit = self.time_cache + intDuration
        self.time_counter = 0
        # ###############################################################

        best = ''
        moves = self.evaluatedMoves()
        depth = 0
        while True:
            # Start at depth = 1
            depth += 1
            # Reset upper and lower bounds
            alpha = float('-inf')
            beta = float('inf')
            # Reset candidate move
            candidate = ''
            # Search for best candidate at current depth
            for move in moves:
                self.move(move)
                temp = -self.alphabeta(depth - 1, -beta, -alpha)
                self.undo()
                if temp > alpha:
                    candidate = move
                    alpha = temp
            # If we ran out of time at this search depth, don't save the current candidate.
            if self.time_cache > self.time_limit:
                break
            # We still have enough time so save the candidate as the best candidate found so far.
            best = candidate
        # Show current player, move number, depth of search, and time elapsed during search.
        self.print_stats_for_move(depth, start)
        return best

    def negascout(self, depth, alpha, beta):  # Not used because I didn't implement T-Tables or better move ordering
        if (depth == 0) or (self.winner() != '?'):
            return self.eval()
        score = float('-inf')

        moves = self.evaluatedMoves()
        for move in moves:
            self.move(move)
            if moves.index(move) == 0:
                score = max(score, -self.negascout(depth - 1, -beta, -alpha))
            else:
                temp = -self.negascout(depth - 1, -alpha - 1, -alpha)
                if temp > alpha and temp < beta:
                    temp = -self.negascout(depth - 1, -beta, -alpha)
                score = max(score, temp)
            self.undo()
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return score

    def allocateTime(self, intDuration):
        if intDuration >= 7000:
            time_left_for_one_move = 7000
        else:
            time_left_for_one_move = intDuration
        return time_left_for_one_move

    def calculateMobility(self):
        current = self.current_player
        if current == 'W':
            white_mobility = len(self.allValidMoves())
            self.current_player = 'B'
            black_mobility = len(self.allValidMoves())
        else:  # if current == 'B':
            black_mobility = len(self.allValidMoves())
            self.current_player = 'W'
            white_mobility = len(self.allValidMoves())
        self.current_player = current
        return white_mobility, black_mobility


class Square(object):
    def __init__(self, x=None, y=None):
        # X IS COLUMN, Y IS ROW
        if x is None or y is None:
            self.x = 0
            self.y = 0
        elif (x < 0 or x > 4) or (y < 0 or y > 5):
            self.x = 0
            self.y = 0
        else:
            self.x = x
            self.y = y


class Move(object):
    def __init__(self, fromSquare, toSquare):
        self.from_square = fromSquare
        self.to_square = toSquare


# Instance of the game state
my_board = State()


def chess_reset():
    # reset the state of the game / your internal variables - note that this function is highly dependent on your
    # implementation
    my_board.reset()


def chess_boardGet():
    # return the state of the game - one example is given below - note that the state has exactly 40 or 41 characters
    return my_board.getBoard()


def chess_boardSet(strIn):
    # read the state of the game from the provided argument and set your internal variables accordingly - note that
    # the state has exactly 40 or 41 characters
    my_board.setBoard(strIn)


def chess_winner():
    # determine the winner of the current state of the game and return '?' or '=' or 'W' or 'B' - note that we are
    # returning a character and not a string
    return my_board.winner()


def chess_isValid(intX, intY):
    if intX < 0:
        return False

    elif intX > 4:
        return False

    if intY < 0:
        return False

    elif intY > 5:
        return False

    return True


def chess_isEnemy(strPiece):
    # with reference to the state of the game, return whether the provided argument is a piece from the side not on
    # move - note that we could but should not use the other is() functions in here but probably
    return my_board.isEnemy(strPiece)


def chess_isOwn(strPiece):
    # with reference to the state of the game, return whether the provided argument is a piece from the side on move
    # - note that we could but should not use the other is() functions in here but probably
    return my_board.isOwn(strPiece)


def chess_isNothing(strPiece):
    # return whether the provided argument is not a piece / is an empty field - note that we could but should not use
    #  the other is() functions in here but probably
    return my_board.isNothing(strPiece)


def chess_eval():
    # with reference to the state of the game, return the the evaluation score of the side on move - note that
    # positive means an advantage while negative means a disadvantage
    return my_board.eval()


def chess_moves():
    # with reference to the state of the game and return the possible moves - one example is given below - note that
    # a move has exactly 6 characters

    moves = my_board.allValidMoves()
    #     print moves
    return moves


def chess_movesShuffled():
    # with reference to the state of the game, determine the possible moves and shuffle them before returning them-
    # note that you can call the chess_moves() function in here
    return my_board.shuffledMoves()


def chess_movesEvaluated():
    # with reference to the state of the game, determine the possible moves and sort them in order of an increasing
    # evaluation score before returning them - note that you can call the chess_moves() function in here

    return my_board.evaluatedMoves()


def chess_move(strIn):
    # perform the supplied move (for example 'a5-a4\n') and update the state of the game / your internal variables
    # accordingly - note that it advised to do a sanity check of the supplied move
    my_board.move(strIn)


def chess_moveRandom():
    # perform a random move and return it - one example output is given below - note that you can call the
    # chess_movesShuffled() function as well as the chess_move() function in here
    random_move = my_board.randomMove()
    my_board.move(random_move)
    return random_move


def chess_moveGreedy():
    # perform a greedy move and return it - one example output is given below - note that you can call the
    # chess_movesEvaluated() function as well as the chess_move() function in here
    if my_board.winner() != '?':
        return ''
    greedy_move = my_board.greedy()
    my_board.move(greedy_move)
    return greedy_move


def chess_moveNegamax(intDepth, intDuration):
    # perform a negamax move and return it - one example output is given below - note that you can call the the other
    #  functions in here
    if my_board.winner() != '?':
        return ''
    negamove = my_board.moveNegamax(intDepth, intDuration)
    my_board.move(negamove)

    return negamove


def chess_moveAlphabeta(intDepth, intDuration):
    # perform a alphabeta move and return it - one example output is given below - note that you can call the the
    # other functions in here
    if my_board.winner() != '?':
        return ''
    # If tournament mode
    if intDepth < 0:
        move = my_board.tournamentMoveAlphabeta(my_board.allocateTime(intDuration))
    # Search up to the given depth or within the time limit.
    else:
        move = my_board.moveAlphabeta(intDepth, intDuration)
    # Perform the move
    my_board.move(move)
    return move


def chess_undo():
    # undo the last move and update the state of the game / your internal variables accordingly - note that you need
    # to maintain an internal variable that keeps track of the previous history for this
    my_board.undo()
