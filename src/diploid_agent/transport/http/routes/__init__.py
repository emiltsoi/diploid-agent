"""HTTP route registration functions."""

from .chat import register_chat
from .config import register_config
from .health import register_health
from .mesh import register_mesh
from .models import register_models
from .plans import register_plans
from .plugins import register_plugins
from .runtime import register_runtime
from .sessions import register_sessions
from .skills import register_skills
from .state import register_state
from .webhook import register_webhook

__all__ = [
    "register_chat",
    "register_config",
    "register_health",
    "register_mesh",
    "register_models",
    "register_plans",
    "register_plugins",
    "register_runtime",
    "register_sessions",
    "register_skills",
    "register_state",
    "register_webhook",
]
