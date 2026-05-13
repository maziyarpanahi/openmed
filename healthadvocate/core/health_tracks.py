"""Health tracks — track active health concerns over time."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

# In-memory storage
_tracks: list[dict] = []


def create_track(concern: str, category: str = "general") -> dict:
    """Create a new health track for an active concern."""
    if not concern or not concern.strip():
        return {"error": "Concern description is required."}

    track_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    track = {
        "id": track_id,
        "concern": concern.strip(),
        "category": category.strip(),
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "notes": [],
    }

    _tracks.append(track)
    return track


def update_track(track_id: str, status: str = "", note: str = "") -> dict:
    """Update a health track — change status or add a note."""
    for track in _tracks:
        if track["id"] == track_id:
            if status and status in ("active", "monitoring", "resolved"):
                track["status"] = status
            if note and note.strip():
                track["notes"].append({
                    "text": note.strip(),
                    "added_at": datetime.now(timezone.utc).isoformat(),
                })
            track["updated_at"] = datetime.now(timezone.utc).isoformat()
            return track
    return {"error": f"Track '{track_id}' not found."}


def list_tracks(status: str | None = None) -> list[dict]:
    """List all tracks, optionally filtered by status."""
    if status:
        return [t for t in _tracks if t["status"] == status]
    return list(_tracks)


def get_track(track_id: str) -> dict:
    """Get a single health track by ID."""
    for track in _tracks:
        if track["id"] == track_id:
            return track
    return {"error": f"Track '{track_id}' not found."}


def get_dashboard() -> dict:
    """Get aggregated dashboard view of all health tracks."""
    active = [t for t in _tracks if t["status"] == "active"]
    monitoring = [t for t in _tracks if t["status"] == "monitoring"]
    resolved = [t for t in _tracks if t["status"] == "resolved"]

    # Recent updates (tracks updated in last 24h, simplified)
    all_sorted = sorted(_tracks, key=lambda t: t["updated_at"], reverse=True)
    recent = [
        {"id": t["id"], "concern": t["concern"], "status": t["status"],
         "updated_at": t["updated_at"], "note_count": len(t["notes"])}
        for t in all_sorted[:10]
    ]

    return {
        "active": len(active),
        "monitoring": len(monitoring),
        "resolved": len(resolved),
        "total": len(_tracks),
        "recent_updates": recent,
        "tracks": list(_tracks),
    }
