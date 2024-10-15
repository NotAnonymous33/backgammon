from flask import Flask, render_template, request, jsonify, abort
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import random
from classes.Board import Board

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# app.config["SECRET_KEY"] = "cheeto"
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)
db = SQLAlchemy(app)

board = Board()

@app.route("/api/get_board", methods=["GET"])
def get_board():
    return jsonify(board.convert())

@app.route("/api/reset_board", methods=["POST"])
def reset():
    global board
    board = Board()
    return jsonify(board.convert())

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
    socketio.run(app, debug=True)


    
    
