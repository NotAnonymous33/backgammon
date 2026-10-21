import unittest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app import app, db, rooms, socketio
from models import Game


class ReconnectionTests(unittest.TestCase):
    def setUp(self):
        self.context = app.app_context()
        self.context.push()
        self.original_engine = db.engines[None]
        self.engine = create_engine(
            "sqlite://", poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        db.engines[None] = self.engine
        db.create_all()
        rooms.clear()
        self.clients = []
        self.room = app.test_client().post("/api/new_game").json["room_code"]

    def tearDown(self):
        for client in self.clients:
            if client.is_connected():
                client.disconnect()
        db.session.remove()
        db.engines[None] = self.original_engine
        self.engine.dispose()
        rooms.clear()
        self.context.pop()

    def connect(self, side):
        client = socketio.test_client(app)
        self.clients.append(client)
        return client, self.join(client, side)

    def join(self, client, side):
        client.emit("join_room", {
            "room_code": self.room, "side": side, "playerType": "human",
        })
        return client.get_received()

    def assert_events(self, events, expected):
        names = [event["name"] for event in events]
        self.assertIn(expected, names)
        self.assertNotIn("error", names)

    def test_same_connection_can_rejoin_its_seat(self):
        white, _ = self.connect("white")
        owner = rooms[self.room]["players"]["white"].copy()
        self.assert_events(self.join(white, "white"), "joined_room")
        self.assertEqual(rooms[self.room]["players"]["white"], owner)

    def test_another_connection_cannot_take_an_active_seat(self):
        self.connect("white")
        owner = rooms[self.room]["players"]["white"].copy()
        intruder, events = self.connect("white")
        self.assertIn("error", [event["name"] for event in events])
        self.assertEqual(rooms[self.room]["players"]["white"], owner)
        intruder.emit("roll_dice", {"room_code": self.room})
        self.assertIn("error", [event["name"] for event in intruder.get_received()])

    def test_disconnect_releases_only_that_players_seat(self):
        white, _ = self.connect("white")
        self.connect("black")
        black_owner = rooms[self.room]["players"]["black"].copy()
        white.disconnect()
        self.assertIsNone(rooms[self.room]["players"]["white"])
        self.assertEqual(rooms[self.room]["players"]["black"], black_owner)

    def test_new_connection_can_rejoin_and_roll_without_resetting_board(self):
        white, _ = self.connect("white")
        game = Game.query.filter_by(room_code=self.room).one()
        positions = game.positions[:]
        white.disconnect()
        replacement, events = self.connect("white")
        self.assert_events(events, "joined_room")
        replacement.emit("roll_dice", {"room_code": self.room})
        self.assert_events(replacement.get_received(), "update_dice")
        db.session.refresh(game)
        self.assertTrue(game.rolled)
        self.assertEqual(game.positions, positions)
        self.assertEqual(game.turn, 1)

    def test_reconnection_preserves_an_existing_roll(self):
        white, _ = self.connect("white")
        white.emit("roll_dice", {"room_code": self.room})
        self.assert_events(white.get_received(), "update_dice")
        game = Game.query.filter_by(room_code=self.room).one()
        dice = game.dice
        white.disconnect()
        _, events = self.connect("white")
        self.assert_events(events, "update_board")
        board = next(event["args"][0] for event in events if event["name"] == "update_board")
        self.assertTrue(board["rolled"])
        self.assertEqual(board["dice"], dice)

    def test_player_can_reclaim_seat_after_server_forgets_rooms(self):
        white, _ = self.connect("white")
        rooms.clear()
        self.assert_events(self.join(white, "white"), "joined_room")
        white.emit("roll_dice", {"room_code": self.room})
        self.assert_events(white.get_received(), "update_dice")


if __name__ == "__main__":
    unittest.main()
