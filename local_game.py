from chess import State

print 'You are playing as white. Enter moves in this format: source-dest.'
print 'Example: a2-a3 moves the leftmost pawn.'
print 'Invalid move strings or hitting enter will have the player make a move for you.'

my_board = State()
print my_board.getBoard()
while my_board.winner() == '?':
    # Player 1
    print 'white move: ',
    my_move = str(raw_input()) + '\n'
    if len(my_move) == 6:
        my_board.move(my_move)
    else:
        alpha_move = my_board.moveAlphabeta(3, 3000)
        print 'Alpha-beta search picked: %s' % alpha_move
        my_board.move(alpha_move)

    print my_board.getBoard()

    # Player 2
    print 'black move: '
    my_board.move(my_board.moveAlphabeta(3, 3000))
    print my_board.getBoard()

print my_board.winner()
