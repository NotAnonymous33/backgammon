from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Game(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    board = db.Column(db.PickleType, nullable=False)
    dice = db.Column(db.PickleType, nullable=False)
    turn = db.Column(db.String(50), nullable=False)