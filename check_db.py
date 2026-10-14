import sqlite3

db = sqlite3.connect("chats.db")
rows = db.execute("SELECT session_id, role, content FROM messages ORDER BY id").fetchall()
print(f"{len(rows)} messages saved\n")
for session_id, role, content in rows:
    print(f"[{session_id[:8]}] {role}: {content[:70]}")