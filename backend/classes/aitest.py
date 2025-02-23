from Board import Board
from agents.FirstAgent import FirstAgent
from agents.RandomAgent import RandomAgent

board = Board()
white = FirstAgent()
black = RandomAgent()
count = 0

while not board.game_over:
    if board.turn.value == 1:
        agent = white
    else:
        agent = black    
    print(board)
    dice, invdice, moves, = board.roll_dice()
    print(dice)
    print(len(moves))
    move = agent.select_move(board)
    board.move_from_sequence(move) 
    count += 1
    print(count)
    if count > 1000:
        print("hi")

if board.white_off == 15:
    print("White wins!")
else:
    print("Black wins!")

    