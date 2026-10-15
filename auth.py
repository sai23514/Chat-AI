"""Login and accounts (step 11).

- Passwords are hashed with scrypt (built into Python), never stored as plain text.
- After login the browser gets an "httponly" cookie that JavaScript can't read.
- Each chat (session_id) belongs to one user; others get "Chat not found".
No extra packages needed.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() != "false"    # false = no login (local testing)
ALLOW_SIGNUP = os.getenv("ALLOW_SIGNUP", "true").lower() != "false"    # false = nobody new can sign up
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"  # true once the site runs on https
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "30"))
COOKIE = "chatai_session"
LOCAL_USER = {"id": 0, "email": "local"}

_db = sqlite3.connect(Path(os.getenv("DB_PATH", "chats.db")), check_same_thread=False)
_db.executescript("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS logins (
    token TEXT PRIMARY KEY,           -- sha256 of the cookie value
    user_id INTEGER NOT NULL,
    expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS chat_owners (
    session_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL);
""")
_db.commit()
_lock = threading.Lock()


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${key.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, salt, key = stored.split("$")
        test = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1, dklen=32)
        return hmac.compare_digest(test.hex(), key)
    except ValueError:
        return False


# ------------------------------------------------------------------ login cookie
def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_login(user_id: int, response: Response) -> None:
    token = secrets.token_urlsafe(32)
    with _lock:
        _db.execute("DELETE FROM logins WHERE expires < ?", (time.time(),))  # tidy old logins
        _db.execute("INSERT INTO logins VALUES (?, ?, ?)",
                    (_token_hash(token), user_id, time.time() + SESSION_DAYS * 86400))
        _db.commit()
    response.set_cookie(COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True,
                        samesite="lax", secure=COOKIE_SECURE)


def user_from_request(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    with _lock:
        row = _db.execute(
            "SELECT users.id, users.email FROM logins JOIN users ON users.id = logins.user_id "
            "WHERE logins.token = ? AND logins.expires > ?", (_token_hash(token), time.time())).fetchone()
    return {"id": row[0], "email": row[1]} if row else None


def current_user(request: Request) -> dict[str, Any]:
    """Add `user: dict = Depends(auth.current_user)` to an endpoint to require login."""
    if not AUTH_ENABLED:
        return LOCAL_USER
    user = user_from_request(request)
    if not user:
        raise HTTPException(401, "Please log in.")
    return user


# ------------------------------------------------------------------ chat ownership
def own_chat(session_id: str, user: dict[str, Any]) -> None:
    """The first user to use a chat owns it. Anyone else gets 404."""
    if not AUTH_ENABLED:
        return
    with _lock:
        row = _db.execute("SELECT user_id FROM chat_owners WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            _db.execute("INSERT INTO chat_owners VALUES (?, ?)", (session_id, user["id"]))
            _db.commit()
        elif row[0] != user["id"]:
            raise HTTPException(404, "Chat not found.")


def forget_chat(session_id: str) -> None:
    with _lock:
        _db.execute("DELETE FROM chat_owners WHERE session_id = ?", (session_id,))
        _db.commit()


# ------------------------------------------------------------------ too many wrong passwords
MAX_FAILS, FAIL_WINDOW = 5, 15 * 60  # 5 wrong tries per 15 minutes per email
_fails: dict[str, list[float]] = {}


def _too_many_fails(email: str) -> bool:
    now = time.time()
    _fails[email] = [t for t in _fails.get(email, []) if now - t < FAIL_WINDOW]
    return len(_fails[email]) >= MAX_FAILS


# ------------------------------------------------------------------ endpoints
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Credentials(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def check_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not EMAIL.match(value):
            raise ValueError("Enter a valid email address")
        return value


router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me")
def me(request: Request):
    """Who is logged in (user is null when nobody is)."""
    user = user_from_request(request) if AUTH_ENABLED else LOCAL_USER
    return {"user": user, "auth_enabled": AUTH_ENABLED, "allow_signup": ALLOW_SIGNUP}


@router.post("/signup")
def signup(creds: Credentials, response: Response):
    if not AUTH_ENABLED:
        raise HTTPException(400, "Login is turned off (AUTH_ENABLED=false).")
    if not ALLOW_SIGNUP:
        raise HTTPException(403, "Sign-up is closed. Ask the owner for an account.")
    try:
        with _lock:
            cur = _db.execute("INSERT INTO users (email, password, created) VALUES (?, ?, ?)",
                              (creds.email, hash_password(creds.password), time.time()))
            _db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists. Log in instead.") from None
    start_login(cur.lastrowid, response)
    return {"user": {"id": cur.lastrowid, "email": creds.email}}


@router.post("/login")
def login(creds: Credentials, response: Response):
    if not AUTH_ENABLED:
        raise HTTPException(400, "Login is turned off (AUTH_ENABLED=false).")
    if _too_many_fails(creds.email):
        raise HTTPException(429, "Too many wrong tries. Wait 15 minutes and try again.")
    with _lock:
        row = _db.execute("SELECT id, password FROM users WHERE email = ?", (creds.email,)).fetchone()
    if not row or not check_password(creds.password, row[1]):
        _fails.setdefault(creds.email, []).append(time.time())
        raise HTTPException(401, "Wrong email or password.")
    _fails.pop(creds.email, None)
    start_login(row[0], response)
    return {"user": {"id": row[0], "email": creds.email}}


@router.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE)
    if token:
        with _lock:
            _db.execute("DELETE FROM logins WHERE token = ?", (_token_hash(token),))
            _db.commit()
    response.delete_cookie(COOKIE)
    return {"ok": True}