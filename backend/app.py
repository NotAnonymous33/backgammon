from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from classes.Board import Board
from models import Game, db
from random import randint

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# app.config["SECRET_KEY"] = "cheeto"
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)


db.init_app(app)

board = Board()

rooms = {}

def generate_code():
    code = "".join(list(map(lambda x: chr(randint(65, 90)), range(5))))
    while code in rooms:
        code = "".join(list(map(lambda x: chr(randint(65, 90)), range(5))))
    
    return code

@app.route("/api/new_game", methods=["POST"])
def new_game():
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
        rolled=board_dict["rolled"]
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

@app.route("/testing", methods=["GET"])
def testing():
    return {"message": "testing"}

@socketio.on("join_room")
def join(data):
    room_code = data["roomCode"]  # TODO: change dictionary keys to gets
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    if not db_board:
        emit("message", {"message": "Room not found"}, room=request.sid)
    else:
        board = Board(board_db=db_board)
        emit("message", {"message": "attempting to join room"})
        join_room(room_code)
        emit("joined_room", {"room_code": room_code}, room=request.sid)
        emit("update_board", board.convert(), room=request.sid)
        
        
@socketio.on("leave_room")
def leave_room(data):
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
def test():
    socketio.emit("message", {"message": "test"})
    return {"message": "test"}

@app.route("/api/button_test", methods=["GET"])
def button_test():
    return {"message": "button was pressed"}

@socketio.on("reset_board")
def reset_board(data):
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
def move(data):
    room_code = data["roomCode"]
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    if not db_board:
        emit("error", {"message": "Room not found"}, room=request.sid)
    else:
        board = Board(board_db=db_board)
        if not board.move_from_sequence(data["moveSequence"]):
            emit('error', {'message': 'Invalid move'}, room=request.sid)
        board_dict = board.convert()
        update_board_db(room_code, board_dict)
        emit("update_board", board_dict, broadcast=True)

@socketio.on("roll_dice")
def roll_dice(data):
    room_code = data["roomCode"]
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    if not db_board:
        emit("error", {"message": "Room not found"}, room=request.sid)
    else:
        board = Board(board_db=db_board)
        valid_dice, invalid_dice, valid_moves = board.roll_dice()
        print(f"{valid_dice=}, {invalid_dice=}")
        update_board_db(room_code, board.convert())
        update_data = {
            "dice": valid_dice,
            "invalidDice": invalid_dice,
            "validMoves": valid_moves
        }
        emit("update_dice", update_data, room=room_code)


@app.route("/api/set_board", methods=["POST"])
def set_board():
    data = request.json
    room_code = data["roomCode"]
    db_board = Game.query.filter_by(room_code=room_code).order_by(Game.id.desc()).first()
    if not db_board:
        return {"status": "error", "message": "Room not found"} 
    else:
        board = Board()
        board.set_board(data['board'])
        board_dict = board.convert()
        add_board_db(room_code, board_dict)
        socketio.emit("update_board", board_dict, room=room_code)
        return {"status": "success"}
        



if __name__ == "__main__":    
    with app.app_context():
        db.create_all()    
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)


    
    
