from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Game(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    board = db.Column(db.PickleType, nullable=False)
    dice = db.Column(db.String(4), nullable=False)
    turn = db.Column(db.Integer, nullable=False)
    white_bar = db.Column(db.Integer, nullable=False)
    black_bar = db.Column(db.Integer, nullable=False)