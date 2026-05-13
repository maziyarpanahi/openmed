"""Family health tracker — manage health profiles for family members."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

# In-memory storage
_family_profiles: dict[str, dict] = {}


def create_profile(name: str, relationship: str) -> dict:
    """Create a new family member health profile."""
    if not name or not name.strip():
        return {"error": "Name is required."}

    profile_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()

    profile = {
        "id": profile_id,
        "name": name.strip(),
        "relationship": relationship.strip() if relationship else "self",
        "conditions": [],
        "medications": [],
        "allergies": [],
        "notes": "",
        "created_at": now,
        "updated_at": now,
    }

    _family_profiles[profile_id] = profile
    return profile


def get_profile(profile_id: str) -> dict:
    """Get a family member's profile by ID."""
    if profile_id not in _family_profiles:
        return {"error": f"Profile '{profile_id}' not found."}
    return _family_profiles[profile_id]


def add_condition(profile_id: str, condition: str) -> dict:
    """Add a condition to a family member's profile."""
    if profile_id not in _family_profiles:
        return {"error": f"Profile '{profile_id}' not found."}
    if not condition or not condition.strip():
        return {"error": "Condition is required."}

    profile = _family_profiles[profile_id]
    profile["conditions"].append({
        "name": condition.strip(),
        "added_at": datetime.now(timezone.utc).isoformat(),
    })
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    return profile


def add_medication(profile_id: str, medication: str, dosage: str = "") -> dict:
    """Add a medication to a family member's profile."""
    if profile_id not in _family_profiles:
        return {"error": f"Profile '{profile_id}' not found."}
    if not medication or not medication.strip():
        return {"error": "Medication name is required."}

    profile = _family_profiles[profile_id]
    profile["medications"].append({
        "name": medication.strip(),
        "dosage": dosage.strip() if dosage else "",
        "added_at": datetime.now(timezone.utc).isoformat(),
    })
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    return profile


def add_allergy(profile_id: str, allergy: str) -> dict:
    """Add an allergy to a family member's profile."""
    if profile_id not in _family_profiles:
        return {"error": f"Profile '{profile_id}' not found."}
    if not allergy or not allergy.strip():
        return {"error": "Allergy is required."}

    profile = _family_profiles[profile_id]
    profile["allergies"].append(allergy.strip())
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    return profile


def update_notes(profile_id: str, notes: str) -> dict:
    """Update notes for a family member's profile."""
    if profile_id not in _family_profiles:
        return {"error": f"Profile '{profile_id}' not found."}

    profile = _family_profiles[profile_id]
    profile["notes"] = notes
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    return profile


def list_profiles() -> list[dict]:
    """List all family member profiles."""
    return list(_family_profiles.values())


def get_family_summary() -> dict:
    """Get an aggregated summary of all family profiles."""
    profiles = list(_family_profiles.values())

    all_conditions = set()
    all_medications = set()
    for profile in profiles:
        for cond in profile.get("conditions", []):
            all_conditions.add(cond["name"].lower())
        for med in profile.get("medications", []):
            all_medications.add(med["name"].lower())

    # Find shared conditions
    condition_owners: dict[str, list[str]] = {}
    for profile in profiles:
        for cond in profile.get("conditions", []):
            name_lower = cond["name"].lower()
            condition_owners.setdefault(name_lower, []).append(profile["name"])

    shared_conditions = {
        cond: owners for cond, owners in condition_owners.items() if len(owners) > 1
    }

    return {
        "total_members": len(profiles),
        "total_conditions": len(all_conditions),
        "total_medications": len(all_medications),
        "shared_conditions": shared_conditions,
        "members": [{"id": p["id"], "name": p["name"], "relationship": p["relationship"],
                      "condition_count": len(p.get("conditions", [])),
                      "medication_count": len(p.get("medications", []))}
                     for p in profiles],
    }


def format_family_context(profile: dict) -> str:
    if "error" in profile:
        return ""
    conditions = ", ".join(c["name"] for c in profile.get("conditions", [])) or "none"
    meds = ", ".join(
        f"{m['name']} ({m['dosage']})" if m.get("dosage") else m["name"]
        for m in profile.get("medications", [])
    ) or "none"
    allergies = ", ".join(profile.get("allergies", [])) or "none"
    notes = profile.get("notes", "") or "none"
    return (
        f"Patient Profile ({profile['name']}, {profile['relationship']}):\n"
        f"- Known Conditions: {conditions}\n"
        f"- Current Medications: {meds}\n"
        f"- Known Allergies: {allergies}\n"
        f"- Notes: {notes}\n"
        "Consider this profile when checking for interactions or relevant advice."
    )
