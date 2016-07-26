from chess import State

my_board = State()
print my_board.getBoard()
while my_board.winner() == '?':
    # Player 1
    print 'white move: ',
    my_move = str(raw_input()) + '\n'

    print 'move string is %d chars long' % len(my_move)

    if len(my_move) == 6:
        my_board.move(my_move)
    else:
        random_move = my_board.moveAlphabeta()
        print 'picking random valid move %s' % random_move
        my_board.move(random_move)

    print my_board.getBoard()

    # Player 2
    print 'black move: '
    my_board.move(my_board.moveAlphabeta())
    print my_board.getBoard()

print my_board.winner()
