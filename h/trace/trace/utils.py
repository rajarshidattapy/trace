"""Utility functions for TRACE."""

import uuid
from datetime import datetime


def generate_episode_id() -> str:
    """Generate unique episode ID."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Get current timestamp in ISO8601 format."""
    return datetime.now().isoformat()


def validate_action(action_type: str, target, value) -> bool:
    """Validate action tuple."""
    valid_actions = {
        "inspect_logs", "inspect_metrics", "inspect_alert",
        "restart_service", "scale_workers", "restart_database",
        "rollback_release", "clear_queue",
        "declare_healthy", "declare_unfixable"
    }
    
    return action_type in valid_actions
