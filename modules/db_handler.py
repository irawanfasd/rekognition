"""
Penyimpanan histori komunikasi menggunakan SQLite.
Ringan, tidak butuh server DB terpisah, cocok untuk device ARM/STB.
"""
import sqlite3
from datetime import datetime


class DBHandler:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                detected_text TEXT NOT NULL,
                confidence REAL
            )
        """)
        conn.commit()
        conn.close()

    def add_entry(self, detected_text, confidence):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO history (timestamp, detected_text, confidence) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), detected_text, confidence),
        )
        conn.commit()
        conn.close()

    def get_recent(self, limit=50):
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def clear(self):
        conn = self._get_conn()
        conn.execute("DELETE FROM history")
        conn.commit()
        conn.close()
