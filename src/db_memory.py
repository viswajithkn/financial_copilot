import sqlite3
import json

class SQLiteChatMemory:
    def __init__(self, db_path="chat_history.db", session_id="default_user"):
        self.db_path = db_path
        self.session_id = session_id
        self.setup_database()

    def setup_database(self):
        """Creates the memory table if it does not exist yet."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_question TEXT,
                    assistant_response TEXT
                )
            """)
            conn.commit()

    def add_message(self, question, response_payload):
        """Saves a conversation turn into the database."""
        # Convert dict or list results into a clean string string
        if isinstance(response_payload, (dict, list)):
            response_str = json.dumps(response_payload)
        else:
            response_str = str(response_payload)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_memory (session_id, user_question, assistant_response)
                VALUES (?, ?, ?)
            """, (self.session_id, question, response_str))
            conn.commit()

    def get_context_string(self, limit=5):
        """Fetches the most recent turns and formats them as a prompt context string."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Fetch the latest rows sorting by ID
            cursor.execute("""
                SELECT user_question, assistant_response 
                FROM chat_memory 
                WHERE session_id = ?
                ORDER BY id DESC 
                LIMIT ?
            """, (self.session_id, limit))
            rows = cursor.fetchall()

        if not rows:
            return ""

        # We reverse them so they read in correct chronological order (oldest to newest)
        rows.reverse()

        context = "\n=== Past Conversation History (From Database) ===\n"
        for q, r in rows:
            context += f"User: {q}\nSystem Response Data: {r}\n\n"
        context += "=================================================\n"
        return context

    def clear_history(self):
        """Clears the history for this session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_memory WHERE session_id = ?", (self.session_id,))
            conn.commit()
