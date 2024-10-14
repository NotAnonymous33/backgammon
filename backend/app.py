from flask import Flask, render_template, request, jsonify
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

@app.route("/api/reset", methods=["POST"])
def reset():
    board = Board()
    return jsonify(board.convert())

@app.route("/api/move", methods=["POST"])
def move():
    data = request.json
    print(data)
    if not board.move(data["current"], data["next"]):
        return jsonify({"error": "Invalid move"})
    return jsonify(board.convert())


if __name__ == "__main__":
    socketio.run(app, debug=True)


    
    
