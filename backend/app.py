from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from classes.board_cpp import Board # type: ignore
from models import Game, db
from random import randint
from classes.agents.HeuristicAgent import HeuristicAgent
from classes.agents.RandomAgent import RandomAgent
from classes.agents.CMCTS2 import MCTSAgent2 as MCTSAgent  # type: ignore
from classes.agents.NNAgent import FinalNNAgent


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# app.config["SECRET_KEY"] = "cheeto"


socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)


db.init_app(app)

rooms = {}

verbose = True

def generate_code():
    code = "".join(list(map(lambda x: chr(randint(65, 90)), range(5))))
    while code in rooms:
        code = "".join(list(map(lambda x: chr(randint(65, 90)), range(5))))
    return code

@app.route("/api/new_game", methods=["POST"])
def handle_new_game():
    room_code = generate_code()
    board = Board()
    board_dict = board.convert()
    add_board_db(room_code, board_dict)
    return jsonify({"room_code": room_code})

def add_board_db(room_code: str, board_dict: dict):
    db_board = Game(
        room_code=room_code,
        positions=board_dict["positions"],
        dice=board_dict["dice"],
        invalid_dice=board_dict["invalid_dice"],
        turn=board_dict["turn"],
        white_bar=board_dict["white_bar"],
        black_bar=board_dict["black_bar"],
        white_off=board_dict["white_off"],
        black_off=board_dict["black_off"],
        rolled=board_dict["rolled"],
        game_over=board_dict["game_over"]
    )
    db.session.add(db_board)
    db.session.commit()

def update_board_db(room_code: str, board_dict: dict):
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    db_board.positions = board_dict["positions"]
    db_board.dice = board_dict["dice"]
    db_board.turn = board_dict["turn"]
    db_board.white_bar = board_dict["white_bar"]
    db_board.black_bar = board_dict["black_bar"]
    db_board.white_off = board_dict["white_off"]
    db_board.black_off = board_dict["black_off"]
    db_board.rolled = board_dict["rolled"]
    db.session.commit()


@socketio.on("join_room")
def handle_join(data):
    room_code = data["room_code"]
    side = data.get("side")  # expected "white" or "black"
    player_type = data.get("playerType", "human")  # "human" or "ai"
    ai_model = data.get("aiModel", "random")  # for AI players, default to "random"
    
    if room_code not in rooms:
        rooms[room_code] = {"players": {"white": None, "black": None}}
        print(rooms)

    if side in ["white", "black"]:
        if player_type == "human":
            if rooms[room_code]["players"].get(side) is not None and rooms[room_code]["players"][side]["type"] == "human":
                emit("error", {"message": f"{side} side already taken."}, room=request.sid)
                return
            rooms[room_code]["players"][side] = {"type": "human", "sid": request.sid}
        else:  # player_type == "ai"
            rooms[room_code]["players"][side] = {"type": "ai", "ai_model": ai_model}
    else:
        # For spectators, if needed
        pass

    join_room(room_code)
    emit("joined_room", {"room_code": room_code, "side": side, "playerType": player_type}, room=request.sid)

    # Send the current board state
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    if db_board:
        board = Board(board_db=db_board)
        emit("update_board", board.convert(), room=request.sid)
    
    # If the side is an AI, trigger an AI move.
    if player_type == "ai":
        socketio.start_background_task(ai_move, room_code)
        socketio.sleep(1)

     
@socketio.on("leave_room")
def handle_leave_room(data):
    room_code = data["room_code"]
    leave_room(room_code)
    emit("left_room", {"room_code": room_code}, room=request.sid)

@socketio.on('connect')
def handle_connect():
    print(f"{request.sid} connected")
    

@socketio.on('disconnect')
def handle_disconnect():
    print(f"{request.sid} disconnected")
    
@app.route("/api/test", methods=["POST"])
def handle_test():
    socketio.emit("message", {"message": "test"})
    return {"message": "test"}

@app.route("/api/button_test", methods=["GET"])
def button_test():
    return {"message": "button was pressed"}

# TODO: i think this isn't used anywhere
@socketio.on("reset_board")
def handle_reset_board(data):
    room_code = data.get('room_code')
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    if not db_board:
        emit("error", {"message": "Room not found"}, room=request.sid)
    else:
        board = Board()
        board_dict = board.convert()
        update_board_db(room_code, board_dict)
        emit("update_board", board_dict, room=room_code)
        

@socketio.on("move")
def handle_move(data):
    room_code = data["room_code"]
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    if not db_board:
        emit("error", {"message": "Room not found"}, room=request.sid)
        return

    board = Board(board_db=db_board)
    
    # Determine which side is supposed to move.
    current_side = "white" if board.turn == 1 else "black"
    room_info = rooms.get(room_code, {}).get("players", {})
    player_info = room_info.get(current_side)
    
    # Only allow the human controller for the current side to move.
    if not (player_info and player_info.get("type") == "human" and player_info.get("sid") == request.sid):
        print("move is trying to be made by the wrong player")
        emit("error", {"message": "Not your turn or you are not authorized to move this side."}, room=request.sid)
        return

    print(f"moving from sequence {data['moveSequence']=}")
    
    move_result = board.move_from_sequence(data["moveSequence"])
    if move_result == False:
        print("move is false")
        emit("error", {"message": "Invalid move"}, room=request.sid)
        return
    else:
        if move_result == True:
            print("move is true")
            emit("game_over", {"winner": "white" if board.white_off == 15 else "black"}, room=room_code)
        board_dict = board.convert()
        update_board_db(room_code, board_dict)
        print("updated db, emitting update_board")
        emit("update_board", board_dict, room=room_code)
        
    
    # If the next turn belongs to an AI, trigger an AI move.
    next_side = "white" if board.turn == 1 else "black"
    next_player = rooms.get(room_code, {}).get("players", {}).get(next_side)
    if next_player and next_player.get("type") == "ai":
        socketio.start_background_task(ai_move, room_code)
        socketio.sleep(1)

def ai_move(room_code):
    with app.app_context():
        print(room_code)
        # this line is causing the error
        db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
        if not db_board:
            return
        board = Board(board_db=db_board)
        
        # Roll dice if not already rolled.
        if not board.rolled:
            roll_result = board.roll_dice()
            if roll_result == False:
                return
            board_dict = board.convert()
            update_board_db(room_code, board_dict)
            socketio.emit("update_dice", {
                "dice": board.dice,
                "invalidDice": board.invalid_dice,
                "validMoves": board.valid_moves,
                "rolled": board.rolled
            }, room=room_code)
        else:
            verbose and print("Dice has already been rolled")
        
        # Determine current side based on board.turn.
        current_side = "white" if board.turn == 1 else "black"
        room_info = rooms.get(room_code, {}).get("players", {})
        player_info = room_info.get(current_side)
        ai_model = player_info.get("ai_model") if player_info and player_info.get("type") == "ai" else "random"
        
        if not player_info or player_info.get("type") != "ai":
            return
        
        # Instantiate the proper AI.
        if ai_model == "heuristic":
            ai = HeuristicAgent()
        elif ai_model == "random":
            ai = RandomAgent()
        elif ai_model == "mcts":
            ai = MCTSAgent(time_budget=5)
        elif ai_model == "neural":
            ai = FinalNNAgent(checkpoint_path="models/main/main_checkpoint_latest.pt")
            
            
        print("selecting move")
        socketio.sleep(1)
        chosen_sequence = ai.select_move(board)
        if chosen_sequence is None:
            # No valid moves.
            return

        board.move_from_sequence(chosen_sequence)
        print(f"ai move: {chosen_sequence}")
        board_dict = board.convert()
        update_board_db(room_code, board_dict)
        socketio.emit("update_board", board_dict, room=room_code)
        socketio.sleep(1)
        
        # Check for game over.
        if board.has_won():
            board.game_over = True
            socketio.emit("game_over", {"winner": "white" if board.white_off == 15 else "black"}, room=room_code)
            return
        
        # If the next turn is also an AI turn, trigger another move after a brief delay.
        next_side = "white" if board.turn == 1 else "black"
        next_player = rooms.get(room_code, {}).get("players", {}).get(next_side)
        if next_player and next_player.get("type") == "ai":
            socketio.start_background_task(ai_move, room_code)
            socketio.sleep(1)
        


@socketio.on("roll_dice")
def handle_roll_dice(data):
    room_code = data["room_code"]
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    if not db_board:
        emit("error", {"message": "Room not found"}, room=request.sid)
        return
    board = Board(board_db=db_board)
    
    current_side = "white" if board.turn == 1 else "black"
    room_info = rooms.get(room_code, {}).get("players", {})
    player_info = room_info.get(current_side)

    if not (player_info and player_info.get("type") == "human" and player_info.get("sid") == request.sid):
        print("dice is trying to be made by the wrong player")
        emit("error", {"message": "Not your turn or you are not authorized to move this side."}, room=request.sid)
        return

    if board.rolled:
        emit("error", {"message": "Dice has already been rolled"}, room=request.sid)
        return
    
    roll_dice_result = board.roll_dice()
    if roll_dice_result == False:
        emit("error", {"message": "Game is Over"}, room=request.sid)
        return
    valid_dice, invalid_dice, valid_moves = roll_dice_result
    update_board_db(room_code, board.convert())
    update_data = {
        "dice": valid_dice,
        "invalidDice": invalid_dice,
        "validMoves": valid_moves,
        "rolled": board.rolled
    }
    emit("update_dice", update_data, room=room_code)
    
    # If it is now an AI turn, trigger the AI move.
    next_side = "white" if board.turn == 1 else "black"
    next_player = rooms.get(room_code, {}).get("players", {}).get(next_side)
    if next_player and next_player.get("type") == "ai":
        socketio.start_background_task(ai_move, room_code)
        socketio.sleep(1)



@app.route("/api/set_board", methods=["POST"])
def handle_set_board():
    verbose and print("app:set_board")
    data = request.json
    room_code = data["room_code"]
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    if not db_board:
        return {"status": "error", "message": "Room not found"} 
    else:
        board = Board()
        board.set_board(data['board'])
        board_dict = board.convert()
        add_board_db(room_code, board_dict)
        emit("update_board", board_dict, room=room_code)
        return {"status": "success"}
        



if __name__ == "__main__":    
    with app.app_context():
        db.create_all()    
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)


    
    
