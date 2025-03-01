from Board import Board
from agents.FirstAgent import FirstAgent
from agents.RandomAgent import RandomAgent
from agents.MCTS import BackgammonMCTSAgent
from agents.HeuristicAgent import HeuristicBackgammonAgent
from time import perf_counter

start = perf_counter()
# black = BackgammonMCTSAgent(time_budget=1)
# white = BackgammonMCTSAgent(time_budget=10)
white = BackgammonMCTSAgent(time_budget=1)
# black = BackgammonMCTSAgent(time_budget=3)
black = RandomAgent()
count = 0
N = 10 # 98.9 seconds
# 24.4 simulations per second time3 vs time3
white_win = 0
black_win = 0
simulations = 0
#while True:
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
        # print(board)
        # print(move)
        board.move_from_sequence(move) 
        count += 1
        cur_count += 1

    if board.white_off == 15:
        white_win += 1
    else:
        black_win += 1
    print(white_win, black_win, white_win + black_win)

print("Time: ", (perf_counter() - start))
print("Games played: ", N)
print(f"Simulations per game: {count}")
print("Time per game: ", (perf_counter() - start)/N)
print("Simulations per second: ", (white.mcts.sim_count)/(perf_counter() - start))

print("White wins: ", white_win)
print("Black wins: ", black_win)
print("Avg game length: ", count/N)

    