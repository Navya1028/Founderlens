"""
history_store.py
-----------------
Saves and loads past assessments to a local JSON file, so history persists
across app restarts (Streamlit's session_state alone only lasts while the
app is open — closing the terminal wipes it).

This is intentionally simple: one JSON file, read/rewritten each time.
Fine for a personal project with dozens/hundreds of entries. If this were
a multi-user deployed app, you'd want a real database instead — but for a
single person's local use, a file is simpler and easier to explain.
"""

import json
import os
from datetime import datetime

HISTORY_FILE = "history.json"


def load_history() -> list[dict]:
    """
    Returns a list of past entries, most recent last.
    Each entry: {"name": str, "timestamp": str, "assessment": str}
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # If the file is corrupted/unreadable, don't crash the app —
        # just start fresh rather than blocking the user.
        return []


def save_entry(name: str, assessment: str) -> None:
    """Append one new entry and rewrite the file."""
    history = load_history()
    history.append({
        "name": name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "assessment": assessment,
    })
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def clear_history() -> None:
    """Wipe all saved history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
