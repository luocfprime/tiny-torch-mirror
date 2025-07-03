import logging

import structlog
from rich.console import Console

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

logger = structlog.get_logger()
console = Console()
