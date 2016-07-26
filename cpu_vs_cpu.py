# Tyler Sorg
import sys
import chess

print chess.chess_boardGet()
depth, count, white_wins, black_wins = 2, 0, 0, 0

while count < 100: # Try these new eval scores out for five games against regular piece values.
    while chess.chess_winner() == '?':
        chess.chess_moveAlphabeta(depth, 3000)
    result = chess.chess_winner()
    if result == 'W':
        white_wins += 1
    elif result == 'B':
        black_wins += 1
    chess.chess_reset()
    count += 1

print "White:%d\nBlack:%d\n" % (white_wins,black_wins)
