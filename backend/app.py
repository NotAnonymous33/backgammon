from flask import Flask, render_template, request, jsonify, abort
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import random
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

def commit_board(board_dict: Board):
    db_board = Game(
        board=board_dict["positions"],
        dice=board_dict["dice"],
        turn=board_dict["turn"],
        white_bar=board_dict["white_bar"],
        black_bar=board_dict["black_bar"]
    )
    db.session.add(db_board)
    db.session.commit()
    

@app.route("/api/get_board", methods=["GET"])
def get_board():
    return jsonify(board.convert())

@app.route("/api/reset_board", methods=["POST"])
def reset():
    global board
    board = Board()
    board_dict = board.convert()
    commit_board(board_dict)
    return jsonify(board_dict)

@app.route("/api/move", methods=["POST"])
def move():
    data = request.json
    if not board.move(data["current"], data["next"]):
        abort(403)
    return jsonify(board.convert())

@app.route("/api/roll_dice", methods=["POST"])
def roll_dice():
    return jsonify(board.roll_dice())


if __name__ == "__main__":    
    with app.app_context():
        db.create_all()    
    socketio.run(app, debug=True)


    
    
