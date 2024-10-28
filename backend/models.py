from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Game(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    positions = db.Column(db.PickleType, nullable=False)
    dice = db.Column(db.String(4), nullable=False)
    turn = db.Column(db.Integer, nullable=False)
    white_bar = db.Column(db.Integer, nullable=False)
    black_bar = db.Column(db.Integer, nullable=False)
    white_off = db.Column(db.Integer, nullable=False)
    black_off = db.Column(db.Integer, nullable=False)
    
    def convert(self):
        return {
            "positions": self.positions,
            "dice": self.dice,
            "turn": self.turn,
            "white_bar": self.white_bar,
            "black_bar": self.black_bar,
            "white_off": self.white_off,
            "black_off": self.black_off
        }