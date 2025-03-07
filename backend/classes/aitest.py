from CBoard import Board # type: ignore
# from Board import Board
from agents.FirstAgent import FirstAgent
from agents.RandomAgent import RandomAgent
# from agents.MCTS import BackgammonMCTSAgent # 1v1 117 52 89 82 117
from agents.CMCTS import BackgammonMCTSAgent # type: ignore
from agents.NNAgent import FinalNNAgent, extract_features

from agents.HeuristicAgent import HeuristicBackgammonAgent
from time import perf_counter, sleep


start = perf_counter()
# white = BackgammonMCTSAgent(time_budget=1)
# black = BackgammonMCTSAgent(time_budget=10)
# white = HeuristicBackgammonAgent([1 for i in range(10)])
# black = HeuristicBackgammonAgent([i for i in range(10)])
# black = HeuristicBackgammonAgent()
# white = RandomAgent()
black = RandomAgent()
white = FinalNNAgent("../backgammon_final_model.pt")
# black = FinalNNAgent()
count = 0
N = 10000
# 24.4 simulations per second time3 vs time3
white_win = 0
black_win = 0
simulations = []
#while True:
for i in range(N):
    board = Board()
    cur_count = 0
    while not board.game_over:
        if len(simulations) == 10:
            pass
        if board.turn == 1:
            agent = white
            # simulations.append(white.mcts.sim_count)
        else:
            agent = black
            # simulations.append(black.mcts.sim_count)
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
    if i % 100 == 99:
        print(white_win, black_win, white_win + black_win)

end = perf_counter()
print("Time: ", (end - start))
print("Games played: ", N)
print("Time per game: ", (end - start)/N)
# print("Simulations per second first half: ", (simulations[len(simulations) // 2])/((end - start)/2))
# print("Simulations per second second half: ", (simulations[-1] - simulations[len(simulations) // 2])/((end - start)/2))

print("White wins: ", white_win)
print("Black wins: ", black_win)
print("Avg game length: ", count/N)
    

        