# from CBoard import Board # type: ignore
# from Board import Board
from board_cpp import Board # type: ignore
from agents.FirstAgent import FirstAgent
from agents.RandomAgent import RandomAgent
# from agents.MCTS import BackgammonMCTSAgent # 1v1 117 52 89 82 117
from agents.CMCTS import BackgammonMCTSAgent # type: ignore
# from agents.NNAgent import FinalNNAgent, extract_features

from agents.HeuristicAgent import HeuristicAgent
from time import perf_counter, sleep


start = perf_counter()
# white = BackgammonMCTSAgent(time_budget=1)
# black = BackgammonMCTSAgent(time_budget=1)
# [2**i for i in range(10)] beats [i for i in range(10)] 77% of the time
white = HeuristicAgent([100, 1, 100, 1, 100, 1, 100, 1, 100, 1])
black = HeuristicAgent([1, 100, 1, 100, 1, 100, 1, 100, 1, 100])
# black = HeuristicBackgammonAgent()
# white = RandomAgent()
# black = RandomAgent()
# white = FinalNNAgent("../backgammon_final_model.pt")
# black = FinalNNAgent()
count = 0
N = 1000 # 1000 games 2.1 seconds (500 game a second) cpp
# 24.4 simulations per second time3 vs time3
white_win = 0
black_win = 0
simulations = []
total_diff = 0
with open("simulations1s150.txt", "w") as f:
    pass
#while True:
for i in range(N):
    board = Board()
    cur_count = 0
    while not board.game_over:
        if board.turn == 1:
            agent = white
            # simulations.append(white.mcts.sim_count)
        else:
            agent = black
            # simulations.append(black.mcts.sim_count)
        dice, invdice, moves, = board.roll_dice()
        # print(board)
        move = agent.select_move(board)
        # print(move)
        board.move_from_sequence(move)
        count += 1
        cur_count += 1

    if board.white_off == 15:
        white_win += 1
    else:
        black_win += 1
    diff = board.black_left - board.white_left # positive if white wins
    total_diff += diff
    
    # if True:
    if (i+1) % 10 == 0:
        print(white_win + black_win, white_win, black_win, diff, total_diff)

    with open("simulations3s.txt", "a") as f:
        f.write(f"{white_win + black_win} {white_win} {black_win} {diff} \n")

end = perf_counter()
print("Time: ", (end - start))
print("Games played: ", N)
print("Time per game: ", (end - start)/N)
simulations.sort()
# print("Lowest simulations: ", simulations[1])
# print("First quartile simulations per second: ", simulations[len(simulations)//4])
# print("Median simulations per second: ", simulations[len(simulations) // 2])

print("White wins: ", white_win)
print("Black wins: ", black_win)
print("Avg game length: ", count/N)
    

        