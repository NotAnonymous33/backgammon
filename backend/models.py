from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Game(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    room_code = db.Column(db.String(5), nullable=False, unique=True)
    positions = db.Column(db.PickleType, nullable=False)
    dice = db.Column(db.String(4), nullable=False)
    invalid_dice = db.Column(db.String(4), nullable=False)
    turn = db.Column(db.Integer, nullable=False)
    white_bar = db.Column(db.Integer, nullable=False)
    black_bar = db.Column(db.Integer, nullable=False)
    white_off = db.Column(db.Integer, nullable=False)
    black_off = db.Column(db.Integer, nullable=False)
    rolled = db.Column(db.Boolean, nullable=False)
    game_over = db.Column(db.Boolean, nullable=False)
    
    def convert(self):
        return {
            "positions": self.positions,
            "dice": self.dice,
            "invalid_dice": self.invalid_dice,
            "turn": self.turn,
            "white_bar": self.white_bar,
            "black_bar": self.black_bar,
            "white_off": self.white_off,
            "black_off": self.black_off,
            "rolled": self.rolled,
            "game_over": self.game_over
        }