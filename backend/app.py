from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from classes.Board import Board
from models import Game, db

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# app.config["SECRET_KEY"] = "cheeto"
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)


db.init_app(app)

board = Board()


def add_board_db(board_dict: dict):
    db_board = Game(
        positions=board_dict["positions"],
        dice=board_dict["dice"],
        turn=board_dict["turn"],
        white_bar=board_dict["white_bar"],
        black_bar=board_dict["black_bar"],
        white_off=board_dict["white_off"],
        black_off=board_dict["black_off"],
        rolled=board_dict["rolled"]
    )
    db.session.add(db_board)
    db.session.commit()
    
def update_board_db(board_dict: dict):
    db_board = Game.query.order_by(Game.id.desc()).first()
    db_board.positions = board_dict["positions"]
    db_board.dice = board_dict["dice"]
    db_board.turn = board_dict["turn"]
    db_board.white_bar = board_dict["white_bar"]
    db_board.black_bar = board_dict["black_bar"]
    db_board.white_off = board_dict["white_off"]
    db_board.black_off = board_dict["black_off"]
    db_board.rolled = board_dict["rolled"]
    db.session.commit()


@socketio.on('connect')
def handle_connect():
    print(f"{request.sid} connected")
    global board
    db_board = Game.query.order_by(Game.id.desc()).first()
    print("queried db")
    if db_board is None:
        emit("update_board", board.convert())
    board = Board(board_db=db_board)
    board_dict = board.convert()
    add_board_db(board_dict)
    emit("update_board", board.convert())
    print("sent board")
    

# @socketio.on('connect')
# def handle_connect():
#     print(f"{request.sid} connected")
#     emit("message", {"message": "connected"})
#     print("emitted message")
    

@socketio.on('disconnect')
def handle_disconnect():
    print(f"{request.sid} disconnected")
    
@app.route("/api/test", methods=["POST"])
def test():
    print("emitting a message")
    socketio.emit("message", {"message": "test"})
    print("emitted message")
    return {"message": "test"}


@socketio.on("reset_board")
def reset_board():
    global board
    board = Board()
    board_dict = board.convert()
    update_board_db(board_dict)
    emit("update_board", board_dict, broadcast=True)
    return {"status": "success"}

@socketio.on("move")
def move(data):
    if not board.move(data["current"], data["next"]):
        emit('error', {'message': 'Invalid move'}, room=request.sid)
    board_dict = board.convert()
    update_board_db(board_dict)
    emit("update_board", board_dict, broadcast=True)
    return {"status": "success"}

@socketio.on("roll_dice")
def roll_dice():
    ret = board.roll_dice()
    add_board_db(board.convert())
    emit("update_dice", ret, broadcast=True)
    return {"status": "success"}


@app.route("/api/set_board", methods=["POST"])
def set_board():
    data = request.json
    board.set_board(data)
    board_dict = board.convert()
    add_board_db(board_dict)
    socketio.emit("update_board", board_dict, broadcast=True)
    return {"status": "success"}


if __name__ == "__main__":    
    with app.app_context():
        db.create_all()    
    socketio.run(app, debug=True)


    
    
