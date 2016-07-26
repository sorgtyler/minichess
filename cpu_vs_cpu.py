# Tyler Sorg
import sys
import chess

# print chess.chess_boardGet()
depth, white_wins, black_wins, draws = 2, 0, 0, 0

for count in xrange(0,10):
    while chess.chess_winner() == '?':
        chess.chess_moveAlphabeta(depth, 3000)
    result = chess.chess_winner()
    if result == 'W':
        white_wins += 1
    elif result == 'B':
        black_wins += 1
    else:
        draws += 1
    chess.chess_reset()

print "White:%d\nBlack:%d\nDraws:%d\n" % (white_wins, black_wins, draws)
