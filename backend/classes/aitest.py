from Board import Board
from agents.FirstAgent import FirstAgent
from agents.RandomAgent import RandomAgent


black = FirstAgent()
white = RandomAgent()
count = 0
N = 100

white_win = 0
black_win = 0
for i in range(N):
    board = Board()
    cur_count = 0
    while not board.game_over:
        if board.turn.value == 1:
            agent = white
        else:
            agent = black    
        dice, invdice, moves, = board.roll_dice()
        move = agent.select_move(board)
        board.move_from_sequence(move) 
        count += 1
        cur_count += 1
        if cur_count > 1000:
            print("Game stuck")
            break

    if board.white_off == 15:
        print("White wins!")
        white_win += 1
    else:
        print("Black wins!")
        black_win += 1
    print(white_win, black_win, white_win + black_win)
        
print("White wins: ", white_win)
print("Black wins: ", black_win)
print("Avg game length: ", count/N)

    