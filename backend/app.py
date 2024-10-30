from flask import Flask, request, jsonify, abort
from flask_socketio import SocketIO, emit, join_room, leave_room
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
        black_off=board_dict["black_off"]
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
    db.session.commit()

    

@app.route("/api/get_board", methods=["GET"])
def get_board():
    global board
    db_board = Game.query.order_by(Game.id.desc()).first()
    if db_board is None:
        return jsonify(board.convert())
    board = Board(board_db=db_board)
    board_dict = board.convert()
    add_board_db(board_dict)
    return jsonify(board_dict)

@app.route("/api/reset_board", methods=["POST"])
def reset():
    global board
    board = Board()
    board_dict = board.convert()
    update_board_db(board_dict)
    return jsonify(board_dict)

@app.route("/api/move", methods=["POST"])
def move():
    data = request.json
    if not board.move(data["current"], data["next"]):
        abort(403)
    board_dict = board.convert()
    update_board_db(board_dict)
    return jsonify(board_dict)

@app.route("/api/roll_dice", methods=["POST"])
def roll_dice():
    ret = board.roll_dice()
    add_board_db(board.convert())
    return jsonify(ret)

@app.route("/api/set_board", methods=["POST"])
def set_board():
    data = request.json
    board.set_board(data)
    board_dict = board.convert()
    add_board_db(board_dict)
    return jsonify(board_dict)


if __name__ == "__main__":    
    with app.app_context():
        db.create_all()    
    socketio.run(app, debug=True)


    
    
