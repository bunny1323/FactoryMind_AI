import os
import sqlite3
import hashlib
import json
import datetime
import logging

logger = logging.getLogger("factorymind")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "factorymind.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    """Returns SHA256 hash of password with a static salt for local sandbox security."""
    salt = "factorymind_salt_123"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def init_db():
    """Creates database tables if they do not exist and populates initial user credentials."""
    logger.info(f"Initializing database at: {DB_PATH}")
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            display_name TEXT,
            role TEXT,
            created_at TEXT
        )
    """)

    # 2. Documents Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            machine TEXT,
            uploaded_by TEXT,
            status TEXT,
            uploaded_at TEXT
        )
    """)

    # 3. Chat Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            created_at TEXT
        )
    """)

    # 4. Chat Messages Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            text TEXT,
            timestamp TEXT,
            evidence TEXT
        )
    """)

    # 5. Machine History (Telemetry logs) Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS machine_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine TEXT,
            date TEXT,
            issue TEXT,
            action TEXT,
            downtime REAL,
            prob REAL
        )
    """)

    conn.commit()

    # Prepopulate default credentials if empty
    cursor.execute("SELECT COUNT(*) as count FROM users")
    if cursor.fetchone()["count"] == 0:
        logger.info("Database empty. Prepopulating default user credentials.")
        now = datetime.datetime.utcnow().isoformat()
        
        # Admin: onepiece / luffy
        cursor.execute(
            "INSERT INTO users (username, password_hash, display_name, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("onepiece", hash_password("luffy"), "Luffy", "admin", now)
        )
        
        # User: zoro / swordsman
        cursor.execute(
            "INSERT INTO users (username, password_hash, display_name, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("zoro", hash_password("swordsman"), "Zoro", "user", now)
        )
        conn.commit()

    # Prepopulate mechatronics logs if empty
    cursor.execute("SELECT COUNT(*) as count FROM machine_history")
    if cursor.fetchone()["count"] == 0:
        logger.info("Prepopulating dynamic mechatronics log history...")
        logs = [
            ("M101", "2026-01-15", "Hydraulic pump temperature high (358 K)", "Cleaned hydraulic radiator and replaced hydraulic oil filters", 3.5, 0.12),
            ("M101", "2026-03-22", "Minor hydraulic pump casing vibration (0.11 mm)", "Inspected coupling and tightened engine-pump flange mounting bolts", 2.0, 0.25),
            ("M101", "2026-05-12", "Vibration alarm E-VIB01 triggered on main pump", "Inspected mounting brackets and aligned pump shaft to engine flywheel", 5.0, 0.72),
            ("M102", "2026-02-10", "Lower engine speed output detected", "Replaced secondary fuel filter element and cleared water separator", 1.5, 0.05)
        ]
        cursor.executemany(
            "INSERT INTO machine_history (machine, date, issue, action, downtime, prob) VALUES (?, ?, ?, ?, ?, ?)",
            logs
        )
        conn.commit()

    conn.close()
    logger.info("Database tables initialized successfully.")

# --- Database Access Helper Functions ---

def get_user_by_username(username: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.lower(),))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def create_user(username: str, password_raw: str, display_name: str, role: str = "user") -> bool:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT INTO users (username, password_hash, display_name, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (username.lower(), hash_password(password_raw), display_name, role, now)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def get_user_sessions(user_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    sessions = cursor.fetchall()
    conn.close()
    return [dict(s) for s in sessions]

def create_chat_session(session_id: str, user_id: str, title: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.utcnow().isoformat()
    cursor.execute(
        "INSERT OR IGNORE INTO chat_sessions (session_id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
        (session_id, user_id, title, now)
    )
    conn.commit()
    conn.close()

def delete_chat_session(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def rename_chat_session(session_id: str, title: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE chat_sessions SET title = ? WHERE session_id = ?", (title, session_id))
    conn.commit()
    conn.close()

def add_chat_message(session_id: str, role: str, text: str, evidence: dict | None = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.utcnow().isoformat()
    evidence_str = json.dumps(evidence) if evidence else ""
    cursor.execute(
        "INSERT INTO chat_messages (session_id, role, text, timestamp, evidence) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, text, now, evidence_str)
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    messages = cursor.fetchall()
    conn.close()
    
    result = []
    for msg in messages:
        d = dict(msg)
        try:
            d["evidence"] = json.loads(d["evidence"]) if d["evidence"] else None
        except Exception as e:
            logger.warning(f"Failed to parse evidence JSON for chat message: {e}", exc_info=True)
            d["evidence"] = None
        result.append(d)
    return result

def get_machine_history_logs(machine_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT date, issue, action, downtime, prob FROM machine_history WHERE machine = ? ORDER BY date DESC", (machine_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Initialize database on module import
init_db()
