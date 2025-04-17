#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <sstream>
#include <set>
#include <map>
#include <utility>
#include <functional>
#include <iostream>

namespace py = pybind11;

class Board
{
public:
    bool rolled;
    int turn;
    int white_off;
    int black_off;
    int white_bar;
    int black_bar;
    int white_left;
    int black_left;
    bool passed;
    bool game_over;
    std::vector<int> positions;
    std::vector<int> dice;
    std::vector<int> invalid_dice;
    std::vector<std::vector<std::pair<int, int>>> valid_moves;
    bool verbose;

    Board(py::dict board_dict = py::dict(), py::object board_db = py::none(), py::object copy = py::none(), bool verbose = false)
        : verbose(verbose)
    {
        if (!copy.is_none())
        {
            Board *src = copy.cast<Board *>();
            positions = src->positions;
            dice = src->dice;
            invalid_dice = src->invalid_dice;
            valid_moves = src->valid_moves;
            rolled = src->rolled;
            turn = src->turn;
            white_off = src->white_off;
            black_off = src->black_off;
            white_bar = src->white_bar;
            black_bar = src->black_bar;
            white_left = src->white_left;
            black_left = src->black_left;
            passed = src->passed;
            game_over = src->game_over;
            return;
        }
        if (!board_db.is_none())
        {
            try
            {
                positions = board_db.attr("positions").cast<std::vector<int>>();
                std::string dice_str = board_db.attr("dice").cast<std::string>();
                dice.clear();
                for (char c : dice_str)
                {
                    dice.push_back(c - '0');
                }
                rolled = board_db.attr("rolled").cast<bool>();
                turn = board_db.attr("turn").cast<int>();
                white_bar = board_db.attr("white_bar").cast<int>();
                black_bar = board_db.attr("black_bar").cast<int>();
                white_off = board_db.attr("white_off").cast<int>();
                black_off = board_db.attr("black_off").cast<int>();
                game_over = board_db.attr("game_over").cast<bool>();
                invalid_dice = get_invalid_dice();
                set_valid_moves();
                white_left = calc_white_left();
                black_left = calc_black_left();
                passed = has_passed();
            }
            catch (const std::exception &)
            {
                initialize_new_board();
            }
            return;
        }
        if (!board_dict.empty())
        {
            try
            {
                positions = board_dict["positions"].cast<std::vector<int>>();
                std::string dice_str = board_dict["dice"].cast<std::string>();
                dice.clear();
                for (char c : dice_str)
                {
                    dice.push_back(c - '0');
                }
                rolled = board_dict["rolled"].cast<bool>();
                turn = board_dict["turn"].cast<int>();
                white_bar = board_dict["white_bar"].cast<int>();
                black_bar = board_dict["black_bar"].cast<int>();
                white_off = board_dict["white_off"].cast<int>();
                black_off = board_dict["black_off"].cast<int>();
                set_valid_moves();
                invalid_dice = get_invalid_dice();
                white_left = calc_white_left();
                black_left = calc_black_left();
                passed = has_passed();
                game_over = board_dict["game_over"].cast<bool>();
            }
            catch (const std::exception &)
            {
                initialize_new_board();
            }
            return;
        }
        initialize_new_board();
    }

    void initialize_new_board()
    {
        positions.resize(24, 0);

        // Initial white positions
        positions[0] = 2;
        positions[11] = 5;
        positions[16] = 3;
        positions[18] = 5;

        // Initial black positions
        positions[5] = -5;
        positions[7] = -3;
        positions[12] = -5;
        positions[23] = -2;

        dice.clear();
        invalid_dice.clear();
        valid_moves.clear();
        rolled = false;
        turn = 1;

        white_off = 0;
        black_off = 0;

        white_bar = 0;
        black_bar = 0;

        white_left = calc_white_left();
        black_left = calc_black_left();
        passed = has_passed();
        game_over = false;
    }

    py::object __deepcopy__(py::dict memo)
    {
        // Create a new Board
        Board *new_board = new Board();
        // Copy all state from this board to the new one
        new_board->copy_state_from(*this);
        // Return the new board with proper ownership management
        return py::cast(new_board, py::return_value_policy::take_ownership);
    }

    std::string __str__() const
    {
        std::stringstream ss;

        // Helper lambda for formatting position
        auto format_position = [](int position) -> std::string
        {
            if (position == 0)
                return "  .  ";
            char color = position > 0 ? 'W' : 'B';
            std::stringstream pos_ss;
            pos_ss << " " << color << std::abs(position);
            return pos_ss.str();
        };

        // Turn info (moved to the beginning to match Python)
        ss << "\n\nTurn: " << (turn == 1 ? "White" : "Black") << "-----------------------------------------------\n";

        // Middle section (bar)
        ss << "\n\nBar:\nWhite: " << white_bar << ", Black: " << black_bar << "\n";

        // Bottom row (positions 11-0, reversed)
        ss << " 12  11  10   9   8   7 |  6   5   4   3   2   1 \n";
        ss << "-------------------------------------------------\n";
        for (int i = 11; i >= 6; i--)
        {
            ss << format_position(positions[i]);
        }
        ss << " |";
        for (int i = 5; i >= 0; i--)
        {
            ss << format_position(positions[i]);
        }

        // Top row (positions 12-23)
        ss << "\n\n 13  14  15  16  17  18 | 19  20  21  22  23  24 \n";
        ss << "-------------------------------------------------\n";
        for (int i = 12; i < 18; i++)
        {
            ss << format_position(positions[i]);
        }
        ss << " |";
        for (int i = 18; i < 24; i++)
        {
            ss << format_position(positions[i]);
        }

        // Off-board area
        ss << "\n\nOff-board:\nWhite: " << white_off << ", Black: " << black_off;

        // Dice info
        ss << "\n\nDice: [";
        for (size_t i = 0; i < dice.size(); i++)
        {
            if (i > 0)
                ss << ", ";
            ss << dice[i];
        }
        ss << "]";

        ss << "\nInvalid Dice: [";
        for (size_t i = 0; i < invalid_dice.size(); i++)
        {
            if (i > 0)
                ss << ", ";
            ss << invalid_dice[i];
        }
        ss << "]";

        // White and Black left counts
        ss << "\n\nWhite left: " << white_left << ", Black left: " << black_left;

        // End separator
        ss << "\n-----------------------------------------------\n";

        return ss.str();
    }

    static constexpr inline int sign(int x)
    {
        return x < 0 ? -1 : (x > 0 ? 1 : 0);
    }

    std::vector<int> list_diff(const std::vector<int> &a, const std::vector<int> &b)
    {
        std::vector<int> a_copy = a;
        for (int i : b)
        {
            auto it = std::find(a_copy.begin(), a_copy.end(), i);
            if (it != a_copy.end())
            {
                a_copy.erase(it);
            }
        }
        return a_copy;
    }

    void copy_state_from(const Board &board)
    {
        positions = board.positions;
        dice = board.dice;
        invalid_dice = board.invalid_dice;
        valid_moves = board.valid_moves;
        rolled = board.rolled;
        turn = board.turn;
        white_off = board.white_off;
        black_off = board.black_off;
        white_bar = board.white_bar;
        black_bar = board.black_bar;
        white_left = board.white_left;
        black_left = board.black_left;
        passed = board.passed;
        game_over = board.game_over;
    }

    bool verify_permutation(Board &board,
                            const std::vector<int> &remaining_dice,
                            const std::vector<int> &used_dice,
                            int &max_length, int &max_die,
                            std::vector<int> &invalid_dice)
    {
        std::vector<int> used_dice_copy = used_dice;

        // base case: all dice are useable
        if (remaining_dice.empty())
        {
            invalid_dice.clear();
            return true;
        }

        // if you can move more dice than current best, replace
        if (used_dice_copy.size() > max_length)
        {
            max_length = used_dice_copy.size();
            invalid_dice = remaining_dice;
        }

        // if you can move the same number of dice but the max die is greater, replace
        if (used_dice_copy.size() == max_length && !used_dice_copy.empty())
        {
            int current_max = *std::max_element(used_dice_copy.begin(), used_dice_copy.end());
            if (current_max > max_die)
            {
                max_die = current_max;
                invalid_dice = remaining_dice;
            }
        }

        // Optimization for when only one die remains
        if (remaining_dice.size() == 1)
        {
            // reentering move white
            if (board.turn == 1 && board.white_bar)
            {
                for (int i = 0; i < 6; i++)
                {
                    if (board.is_valid(-1, i))
                    {
                        max_die = std::max(remaining_dice[0], max_die);
                        invalid_dice.clear();
                        max_length = used_dice_copy.size() + 1;
                        return true;
                    }
                }
                return false;
            }

            // reentering move black
            if (board.turn == -1 && board.black_bar)
            {
                for (int i = 23; i > 17; i--)
                {
                    if (board.is_valid(-1, i))
                    {
                        max_die = std::max(remaining_dice[0], max_die);
                        invalid_dice.clear();
                        max_length = used_dice_copy.size() + 1;
                        return true;
                    }
                }
                return false;
            }

            // normal moves
            for (int start = 0; start < 24; start++)
            {
                if (sign(board.positions[start]) != board.turn)
                {
                    continue;
                }
                int end = start + remaining_dice[0] * board.turn;
                if (board.is_valid(start, end))
                {
                    max_die = std::max(remaining_dice[0], max_die);
                    invalid_dice.clear();
                    max_length = used_dice_copy.size() + 1;
                    return true;
                }
            }

            // bearing off
            if (board.can_bearoff())
            {
                if (board.turn == 1)
                {
                    for (int i = 18; i < 24; i++)
                    {
                        if (board.is_valid(i, 100))
                        {
                            max_die = std::max(remaining_dice[0], max_die);
                            invalid_dice.clear();
                            max_length = used_dice_copy.size() + 1;
                            return true;
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < 6; i++)
                    {
                        if (board.is_valid(i, -100))
                        {
                            max_die = std::max(remaining_dice[0], max_die);
                            invalid_dice.clear();
                            max_length = used_dice_copy.size() + 1;
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        Board board_copy = board;

        // reentering moves white
        if (board.turn == 1 && board.white_bar)
        {
            for (int i = 0; i < 6; i++)
            {
                if (board.is_valid(-1, i))
                {
                    board_copy.copy_state_from(board);
                    board_copy.move(-1, i, true);
                    std::vector<int> diff = list_diff(board.dice, board_copy.dice);
                    std::vector<int> new_used = used_dice_copy;
                    new_used.insert(new_used.end(), diff.begin(), diff.end());
                    if (verify_permutation(board_copy, board_copy.dice, new_used, max_length, max_die, invalid_dice))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        // reentering moves black
        if (board.turn == -1 && board.black_bar)
        {
            for (int i = 23; i > 17; i--)
            {
                if (board.is_valid(-1, i))
                {
                    board_copy.copy_state_from(board);
                    board_copy.move(-1, i, true);
                    std::vector<int> diff = list_diff(board.dice, board_copy.dice);
                    std::vector<int> new_used = used_dice_copy;
                    new_used.insert(new_used.end(), diff.begin(), diff.end());
                    if (verify_permutation(board_copy, board_copy.dice, new_used, max_length, max_die, invalid_dice))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        // bearing off
        if (board.can_bearoff())
        {
            if (board.turn == 1)
            {
                for (int i = 18; i < 24; i++)
                {
                    if (board.is_valid(i, 100))
                    {
                        board_copy.copy_state_from(board);
                        board_copy.move(i, 100, true);
                        std::vector<int> new_remaining = remaining_dice;
                        new_remaining.erase(new_remaining.begin());
                        std::vector<int> new_used = used_dice_copy;
                        new_used.push_back(remaining_dice[0]);
                        if (verify_permutation(board_copy, new_remaining, new_used, max_length, max_die, invalid_dice))
                        {
                            return true;
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < 6; i++)
                {
                    if (board.is_valid(i, -100))
                    {
                        board_copy.copy_state_from(board);
                        board_copy.move(i, -100, true);
                        std::vector<int> new_remaining = remaining_dice;
                        new_remaining.erase(new_remaining.begin());
                        std::vector<int> new_used = used_dice_copy;
                        new_used.push_back(remaining_dice[0]);
                        if (verify_permutation(board_copy, new_remaining, new_used, max_length, max_die, invalid_dice))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        // normal moves
        for (int start = 0; start < 24; start++)
        {
            if (sign(board.positions[start]) != board.turn)
            {
                continue;
            }
            int end = start + remaining_dice[0] * board.turn;
            if (board.is_valid(start, end))
            {
                board_copy.copy_state_from(board);
                if (board_copy.move(start, end, true))
                {
                    std::vector<int> new_remaining = remaining_dice;
                    new_remaining.erase(new_remaining.begin());
                    std::vector<int> new_used = used_dice_copy;
                    new_used.push_back(remaining_dice[0]);
                    if (verify_permutation(board_copy, new_remaining, new_used, max_length, max_die, invalid_dice))
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    std::vector<int> get_invalid_dice()
    {
        if (verbose)
            std::cout << "Board:get_invalid_dice" << std::endl;

        std::vector<int> invalid_dice_result = dice;
        int max_length = 0;
        int max_die = 0;

        Board board_copy = *this;
        if (verify_permutation(board_copy, dice, {}, max_length, max_die, invalid_dice_result))
        {
            return {};
        }

        // Remove invalid dice from this->dice
        std::vector<int> result = invalid_dice_result;
        for (int die : result)
        {
            auto it = std::find(dice.begin(), dice.end(), die);
            if (it != dice.end())
            {
                dice.erase(it);
            }
        }

        return result;
    }

    std::set<std::pair<int, int>> get_single_moves(int min_point = -1)
    {
        std::set<std::pair<int, int>> moves;
        if (min_point == -1)
        {
            min_point = turn == 1 ? 0 : 23;
        }

        if (turn == 1)
        {
            // reentering checkers
            if (white_bar)
            {
                for (int die : dice)
                {
                    if (is_valid(-1, die - 1))
                    {
                        moves.insert({-1, die - 1});
                    }
                }
                return moves;
            }

            // bearing off
            for (int start = 18; start < 24; start++)
            {
                if (is_valid(start, 100))
                {
                    moves.insert({start, 100});
                }
            }

            // normal moves
            for (int start = min_point; start < 24; start++)
            {
                for (int die : dice)
                {
                    if (is_valid(start, start + die))
                    {
                        moves.insert({start, start + die});
                    }
                }
            }

            return moves;
        }

        // Black's turn
        // reentering checkers
        if (black_bar)
        {
            for (int i = 23; i > 17; i--)
            {
                if (is_valid(-1, i))
                {
                    moves.insert({-1, i});
                }
            }
            return moves;
        }

        // bearing off
        for (int start = 5; start >= 0; start--)
        {
            if (is_valid(start, -100))
            {
                moves.insert({start, -100});
            }
        }

        // normal moves
        for (int start = min_point; start >= 0; start--)
        {
            for (int die : dice)
            {
                if (is_valid(start, start - die))
                {
                    moves.insert({start, start - die});
                }
            }
        }

        return moves;
    }

    std::vector<std::vector<std::pair<int, int>>> dfs(Board &board, std::vector<std::pair<int, int>> prev_moves, int min_point = -1)
    {
        if (min_point == -1)
        {
            min_point = turn == 1 ? 0 : 23;
        }
        if (board.dice.empty())
        {
            if (!prev_moves.empty())
            {
                return {prev_moves};
            }
            return {};
        }

        std::vector<std::vector<std::pair<int, int>>> moves;

        if (board.dice.size() == 1)
        {
            auto single_moves = board.get_single_moves(min_point);
            for (const auto &move : single_moves)
            {
                std::vector<std::pair<int, int>> new_moves = prev_moves;
                new_moves.push_back(move);
                moves.push_back(new_moves);
            }
            return moves;
        }

        auto single_moves = board.get_single_moves(min_point);
        for (const auto &move : single_moves)
        {
            Board board_copy = board;
            board_copy.move(move.first, move.second, true);
            std::vector<std::pair<int, int>> new_prev_moves = prev_moves;
            new_prev_moves.push_back(move);
            auto dfs_results = dfs(board_copy, new_prev_moves, move.first != -1 ? move.first : min_point);
            moves.insert(moves.end(), dfs_results.begin(), dfs_results.end());
        }

        return moves;
    }

    void set_valid_moves()
    {
        if (verbose)
            std::cout << "Board:get_valid_moves" << std::endl;

        valid_moves.clear();

        Board board_copy = *this;
        valid_moves = dfs(board_copy, {});
    }

    bool can_bearoff() const
    {
        if (turn == 1)
        {
            if (white_bar)
            {
                return false;
            }
            for (int i = 0; i < 18; i++)
            {
                if (positions[i] > 0)
                {
                    return false;
                }
            }
        }
        else
        {
            if (black_bar)
            {
                return false;
            }
            for (int i = 6; i < 24; i++)
            {
                if (positions[i] < 0)
                {
                    return false;
                }
            }
        }
        return true;
    }

    py::dict convert() const
    {
        py::dict ret;
        ret["positions"] = positions;
        ret["turn"] = turn;

        std::string dice_str;
        for (int d : dice)
        {
            dice_str += std::to_string(d);
        }
        ret["dice"] = dice_str;

        std::string invalid_dice_str;
        for (int d : invalid_dice)
        {
            invalid_dice_str += std::to_string(d);
        }
        ret["invalid_dice"] = invalid_dice_str;

        ret["white_bar"] = white_bar;
        ret["black_bar"] = black_bar;
        ret["rolled"] = rolled;
        ret["white_off"] = white_off;
        ret["black_off"] = black_off;
        ret["valid_moves"] = valid_moves;
        ret["game_over"] = game_over;

        return ret;
    }

    bool has_passed() const
    {
        if (white_bar > 0 || black_bar > 0)
        {
            return false;
        }

        int lowest_white = 24;
        int highest_black = -1;

        // Find lowest white position
        for (int i = 0; i < 24; i++)
        {
            if (positions[i] > 0)
            {
                lowest_white = i;
                break;
            }
        }

        // Find highest black position
        for (int i = 23; i >= 0; i--)
        {
            if (positions[i] < 0)
            {
                highest_black = i;
                break;
            }
        }

        // If all pieces are borne off for one color
        if (lowest_white == -1 || highest_black == 24)
        {
            return true;
        }

        // Check if all black pieces are past all white pieces
        return lowest_white > highest_black;
    }

    int calc_white_left() const
    {
        int total = 0;
        for (int i = 0; i < 24; i++)
        {
            if (positions[i] > 0)
            {
                total += (24 - i) * positions[i];
            }
        }
        total += white_bar * 24;
        return total;
    }

    int calc_black_left() const
    {
        int total = 0;
        for (int i = 0; i < 24; i++)
        {
            if (positions[i] < 0)
            {
                total += (i + 1) * (-positions[i]);
            }
        }
        total += black_bar * 24;
        return total;
    }

    py::object move_from_sequence(const std::vector<std::pair<int, int>> &sequence)
    {
        if (verbose)
            std::cout << "Board:move_from_sequence" << std::endl;

        if (valid_moves.empty() && sequence.empty())
        {
            swap_turn();
            return py::none();
        }

        // Check if sequence is in valid_moves
        bool valid = false;
        for (const auto &moves : valid_moves)
        {
            if (moves.size() == sequence.size())
            {
                bool match = true;
                for (size_t i = 0; i < moves.size(); i++)
                {
                    if (moves[i] != sequence[i])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                {
                    valid = true;
                    break;
                }
            }
        }

        if (!valid)
        {
            return py::bool_(false);
        }

        // Execute the moves
        for (const auto &move : sequence)
        {
            this->move(move.first, move.second, true);
        }

        if (!passed)
        {
            passed = has_passed();
        }

        white_left = calc_white_left();
        black_left = calc_black_left();

        if (has_won())
        {
            game_over = true;
            return py::bool_(true);
        }

        if (rolled && dice.empty())
        {
            swap_turn();
        }

        return py::none();
    }

    bool has_won() const
    {
        return white_off == 15 || black_off == 15;
    }

    bool move(int current, int next, bool bypass = false)
    {
        if (!bypass && !is_valid(current, next))
        {
            return false;
        }

        // bearing off
        if (next == 100 || next == -100)
        {
            if (turn == 1)
            {
                white_off += 1;
                // Find the die to remove
                auto dice_it = std::find(dice.begin(), dice.end(), 24 - current);
                if (dice_it != dice.end())
                {
                    dice.erase(dice_it);
                }
                else
                {
                    // Remove the max die
                    auto max_it = std::max_element(dice.begin(), dice.end());
                    if (max_it != dice.end())
                    {
                        dice.erase(max_it);
                    }
                }
                positions[current] -= 1;
            }
            else
            {
                black_off += 1;
                // Find the die to remove
                auto dice_it = std::find(dice.begin(), dice.end(), current + 1);
                if (dice_it != dice.end())
                {
                    dice.erase(dice_it);
                }
                else
                {
                    // Remove the max die
                    auto max_it = std::max_element(dice.begin(), dice.end());
                    if (max_it != dice.end())
                    {
                        dice.erase(max_it);
                    }
                }
                positions[current] += 1;
            }
            return true;
        }

        // capturing a piece
        if (turn == 1)
        {
            if (positions[next] == -1)
            {
                positions[next] = 0;
                black_bar += 1;
            }
        }
        else
        {
            if (positions[next] == 1)
            {
                positions[next] = 0;
                white_bar += 1;
            }
        }

        // reentering checkers
        if (current == -1)
        {
            if (turn == 1)
            {
                white_bar -= 1;
                // Remove die from dice
                auto it = std::find(dice.begin(), dice.end(), next + 1);
                if (it != dice.end())
                {
                    dice.erase(it);
                }
                positions[next] += 1;
            }
            else
            {
                black_bar -= 1;
                // Remove die from dice
                auto it = std::find(dice.begin(), dice.end(), 24 - next);
                if (it != dice.end())
                {
                    dice.erase(it);
                }
                positions[next] -= 1;
            }
        }
        else
        { // not reentering
            positions[current] -= turn;
            positions[next] += turn;
            // Remove die from dice
            auto it = std::find(dice.begin(), dice.end(), (next - current) * turn);
            if (it != dice.end())
            {
                dice.erase(it);
            }
        }
        return true;
    }

    void swap_turn()
    {
        if (verbose)
            std::cout << "Board:swap_turn" << std::endl;

        turn = -turn;
        rolled = false;
        dice.clear();
        invalid_dice.clear();
        valid_moves.clear();
    }

    bool is_valid(int current, int next) const
    {
        // current can't be outside valid range
        if (current < -1 || current > 23)
        {
            return false;
        }
        if ((next < 0 || next > 23) && abs(next) != 100)
        {
            return false;
        }

        // bearing off
        if ((next == 100 && turn == 1) || (next == -100 && turn == -1))
        {
            if (!can_bearoff())
            {
                return false;
            }
            if (sign(positions[current]) != turn)
            {
                return false;
            }

            if (turn == 1)
            {
                for (int die : dice)
                {
                    if (die == 24 - current)
                    {
                        return true;
                    }
                }
                // no exact dice
                if (dice.empty() || 24 - current > *std::max_element(dice.begin(), dice.end()))
                {
                    return false;
                }

                for (int pos = 18; pos < 24; pos++)
                {
                    if (positions[pos] > 0)
                    {
                        if (current == pos)
                        {
                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    }
                }
            }
            else
            { // turn == -1
                for (int die : dice)
                {
                    if (die == current + 1)
                    {
                        return true;
                    }
                }

                if (dice.empty() || current + 1 > *std::max_element(dice.begin(), dice.end()))
                {
                    return false;
                }

                for (int pos = 5; pos >= 0; pos--)
                {
                    if (positions[pos] < 0)
                    {
                        if (current == pos)
                        {
                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    }
                }
                return false;
            }
        }

        // reentering checkers
        if (white_bar && turn == 1)
        {
            auto dice_it = std::find(dice.begin(), dice.end(), next + 1);
            if (dice_it == dice.end())
            {
                return false;
            }
            if (current != -1)
            {
                return false;
            }
            if (next > 5)
            {
                return false;
            }
            if (positions[next] < -1)
            {
                return false;
            }
            return true;
        }

        if (black_bar && turn == -1)
        {
            if (current != -1)
            {
                return false;
            }
            if (next < 18)
            {
                return false;
            }
            auto dice_it = std::find(dice.begin(), dice.end(), 24 - next);
            if (dice_it == dice.end())
            {
                return false;
            }
            if (positions[next] > 1)
            {
                return false;
            }
            return true;
        }

        if (sign(positions[current]) != turn)
        {
            return false;
        }
        if (positions[next] * turn < -1)
        {
            return false;
        }

        for (int die : dice)
        {
            if ((next - current) * turn == die)
            {
                return true;
            }
        }
        return false;
    }

    py::object roll_dice()
    {
        if (verbose)
            std::cout << "Board:roll_dice" << std::endl;

        if (game_over)
        {
            return py::bool_(false);
        }

        if (rolled)
        {
            return py::make_tuple(dice, invalid_dice, valid_moves);
        }

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(1, 6);

        dice = {distrib(gen), distrib(gen)};

        if (dice[0] == dice[1])
        {
            dice.push_back(dice[0]);
            dice.push_back(dice[0]);
        }

        rolled = true;
        invalid_dice = get_invalid_dice();
        set_valid_moves();

        return py::make_tuple(dice, invalid_dice, valid_moves);
    }

    py::object set_dice(const std::vector<int> &new_dice)
    {
        if (verbose)
            std::cout << "Board:set_dice" << std::endl;

        if (game_over)
        {
            return py::bool_(false);
        }

        if (rolled)
        {
            return py::make_tuple(dice, invalid_dice, valid_moves);
        }

        dice = new_dice;
        rolled = true;
        invalid_dice = get_invalid_dice();
        set_valid_moves();

        return py::make_tuple(dice, invalid_dice, valid_moves);
    }

    py::dict set_board(py::dict data)
    {
        if (data.contains("positions"))
        {
            positions = data["positions"].cast<std::vector<int>>();
        }
        if (data.contains("dice"))
        {
            std::string dice_str = data["dice"].cast<std::string>();
            dice.clear();
            for (char c : dice_str)
            {
                dice.push_back(c - '0');
            }
        }
        if (data.contains("turn"))
        {
            turn = data["turn"].cast<int>();
        }
        return convert();
    }
};

PYBIND11_MODULE(board_cpp, m)
{
    py::class_<Board>(m, "Board")
        .def(py::init<py::dict, py::object, py::object, bool>(),
             py::arg("board_dict") = py::dict(),
             py::arg("board_db") = py::none(),
             py::arg("copy") = py::none(),
             py::arg("verbose") = false)
        .def("__str__", &Board::__str__)
        .def("get_invalid_dice", &Board::get_invalid_dice)
        .def("get_single_moves", &Board::get_single_moves, py::arg("min_point") = -1)
        .def("set_valid_moves", &Board::set_valid_moves)
        .def("can_bearoff", &Board::can_bearoff)
        .def("convert", &Board::convert)
        .def("has_passed", &Board::has_passed)
        .def("calc_white_left", &Board::calc_white_left)
        .def("calc_black_left", &Board::calc_black_left)
        .def("move_from_sequence", &Board::move_from_sequence)
        .def("has_won", &Board::has_won)
        .def("move", &Board::move, py::arg("current"), py::arg("next"), py::arg("bypass") = false)
        .def("swap_turn", &Board::swap_turn)
        .def("is_valid", &Board::is_valid)
        .def("roll_dice", &Board::roll_dice)
        .def("set_dice", &Board::set_dice)
        .def("set_board", &Board::set_board)
        .def("__deepcopy__", &Board::__deepcopy__, py::arg("memo"))
        .def("clone", [](const Board &self)
             { return new Board(self); }, py::return_value_policy::take_ownership)
        .def_readwrite("rolled", &Board::rolled)
        .def_readwrite("turn", &Board::turn)
        .def_readwrite("white_off", &Board::white_off)
        .def_readwrite("black_off", &Board::black_off)
        .def_readwrite("white_bar", &Board::white_bar)
        .def_readwrite("black_bar", &Board::black_bar)
        .def_readwrite("white_left", &Board::white_left)
        .def_readwrite("black_left", &Board::black_left)
        .def_readwrite("passed", &Board::passed)
        .def_readwrite("game_over", &Board::game_over)
        .def_readwrite("positions", &Board::positions)
        .def_readwrite("dice", &Board::dice)
        .def_readwrite("invalid_dice", &Board::invalid_dice)
        .def_readwrite("valid_moves", &Board::valid_moves)
        .def_readwrite("verbose", &Board::verbose);
}