"""
Security and cost control utilities for ReguSense AI application.
Implements input validation, rate limiting, and cost tracking.
"""

import re
import os
import logging
from html import escape
from typing import Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CostTracker:
    """Track OpenAI API usage and costs."""
    total_cost: float = 0.0
    total_tokens: int = 0
    request_count: int = 0
    daily_cost: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    max_budget: float = 0.50
    
    # Pricing (as of Jan 2026)
    EMBEDDING_COST_PER_1K = 0.00002  # text-embedding-3-small
    GPT4O_INPUT_COST_PER_1K = 0.0025
    GPT4O_OUTPUT_COST_PER_1K = 0.01
    GPT4O_MINI_INPUT_COST_PER_1K = 0.00015
    GPT4O_MINI_OUTPUT_COST_PER_1K = 0.0006
    
    def track_embedding(self, total_tokens: int) -> float:
        """Track embedding API call cost."""
        cost = (total_tokens / 1000) * self.EMBEDDING_COST_PER_1K
        self._update_tracker(cost, total_tokens)
        return cost
    
    def track_gpt4o(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Track GPT-4o API call cost."""
        input_cost = (prompt_tokens / 1000) * self.GPT4O_INPUT_COST_PER_1K
        output_cost = (completion_tokens / 1000) * self.GPT4O_OUTPUT_COST_PER_1K
        total_cost = input_cost + output_cost
        total_tokens = prompt_tokens + completion_tokens
        self._update_tracker(total_cost, total_tokens)
        return total_cost
    
    def track_gpt4o_mini(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Track GPT-4o-mini API call cost."""
        input_cost = (prompt_tokens / 1000) * self.GPT4O_MINI_INPUT_COST_PER_1K
        output_cost = (completion_tokens / 1000) * self.GPT4O_MINI_OUTPUT_COST_PER_1K
        total_cost = input_cost + output_cost
        total_tokens = prompt_tokens + completion_tokens
        self._update_tracker(total_cost, total_tokens)
        return total_cost
    
    def _update_tracker(self, cost: float, tokens: int):
        """Update internal tracking metrics."""
        self.total_cost += cost
        self.total_tokens += tokens
        self.request_count += 1
        
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_cost[today] += cost
        
        # Log if budget exceeded
        if self.total_cost > self.max_budget:
            logger.warning(f"Budget exceeded: ${self.total_cost:.4f} > ${self.max_budget}")
    
    def check_budget(self) -> bool:
        """Check if budget limit has been exceeded."""
        return self.total_cost <= self.max_budget
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            'total_cost': round(self.total_cost, 6),
            'total_tokens': self.total_tokens,
            'request_count': self.request_count,
            'avg_cost_per_request': round(self.total_cost / max(self.request_count, 1), 6),
            'today_cost': round(self.daily_cost[datetime.now().strftime('%Y-%m-%d')], 6),
            'max_budget': self.max_budget,
            'daily_cost': dict(self.daily_cost)
        }


@dataclass
class RateLimiter:
    """Rate limiting for user requests."""
    max_requests: int = 10  # requests per window
    time_window: int = 3600  # 1 hour in seconds
    request_times: list = field(default_factory=list)
    
    def check_rate_limit(self) -> Tuple[bool, int]:
        """Check if rate limit is exceeded.
        
        Returns:
            Tuple of (is_allowed, seconds_until_reset)
        """
        now = datetime.now()
        window_start = now - timedelta(seconds=self.time_window)
        
        # Remove old requests
        self.request_times = [
            req_time for req_time in self.request_times
            if req_time > window_start
        ]
        
        # Check limit
        request_count = len(self.request_times)
        
        if request_count >= self.max_requests:
            if self.request_times:
                reset_time = self.request_times[0] + timedelta(seconds=self.time_window)
                seconds_left = int((reset_time - now).total_seconds())
                return False, max(0, seconds_left)
            return False, 0
        
        return True, 0
    
    def record_request(self):
        """Record a new request."""
        self.request_times.append(datetime.now())


def validate_user_input(query: str, max_length: int = 500) -> Tuple[bool, str]:
    """
    Validate and sanitize user input for security.
    
    Args:
        query: User input query
        max_length: Maximum allowed query length
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Length checks
    if not query or len(query.strip()) == 0:
        return False, "Query cannot be empty"
    
    if len(query) < 3:
        return False, "Query too short (minimum 3 characters)"
    
    if len(query) > max_length:
        return False, f"Query too long (maximum {max_length} characters)"
    
    # Prompt injection detection
    suspicious_patterns = [
        r"\bignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?|rules?|commands?)",
        r"\bignore\s+(previous|all|prior|everything)",
        r"\bforget\s+(everything|all|previous)",
        r"\bdisregard\s+(all|previous|prior)",
        r"\bsystem\s*:",
        r"<\s*script",
        r"javascript\s*:",
        r"(eval|exec)\s*\(",
        r"__import__",
        r"<!--",
        r"--!>",
        r"onerror\s*=",
        r"<\s*img\s+.*onerror",
        r"\{%.*%\}",  # Template injection
        r"\{\{.*\}\}",  # Template injection
        r"\$\{.*\}",  # Template injection ${}
        r"SELECT\s+.*\s+FROM",  # SQL injection attempt
        r"DROP\s+TABLE",  # SQL injection
        r"DELETE\s+FROM",  # SQL injection
        r"UPDATE\s+.*\s+SET",  # SQL injection
        r"(1|')\s*OR\s*('?1'?|'1')\s*=\s*('?1'?|'1')",  # SQL injection patterns
        r"\bUNION\s+SELECT",  # SQL injection
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning(f"Suspicious input pattern detected: {pattern}")
            return False, "Input contains suspicious patterns and has been blocked for security reasons"
    
    # Check for excessive special characters (potential injection)
    special_char_count = len(re.findall(r'[<>&\'";`|$]', query))
    if special_char_count > 10:
        return False, "Input contains too many special characters"
    
    return True, "Valid"


def sanitize_input(query: str) -> str:
    """
    Sanitize user input by escaping HTML and removing control characters.
    
    Args:
        query: User input query
    
    Returns:
        Sanitized query string
    """
    # Remove control characters (except newlines, tabs, carriage returns)
    query = ''.join(char for char in query if ord(char) >= 32 or char in '\n\r\t')
    
    # Escape HTML
    query = escape(query)
    
    # Normalize whitespace (but preserve single newlines)
    lines = query.split('\n')
    normalized_lines = [' '.join(line.split()) for line in lines]
    query = '\n'.join(normalized_lines)
    
    # Remove multiple consecutive newlines
    query = re.sub(r'\n{3,}', '\n\n', query)
    
    return query.strip()


def check_api_keys() -> Tuple[bool, list]:
    """
    Validate that required API keys are present.
    
    Returns:
        Tuple of (all_valid, missing_keys)
    """
    required_keys = [
        'OPENAI_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_INDEX_NAME'
    ]
    
    missing_keys = []
    for key in required_keys:
        value = os.getenv(key)
        if not value or len(value.strip()) == 0:
            missing_keys.append(key)
    
    return len(missing_keys) == 0, missing_keys


def mask_api_key(key: str) -> str:
    """
    Mask API key for safe logging.
    
    Args:
        key: API key to mask
    
    Returns:
        Masked key (first 8 chars + ***)
    """
    if not key or len(key) < 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def log_security_event(event_type: str, details: Dict[str, Any], level: str = "WARNING"):
    """
    Log security-related events.
    
    Args:
        event_type: Type of security event
        details: Event details dictionary
        level: Log level (INFO, WARNING, ERROR, CRITICAL)
    """
    timestamp = datetime.now().isoformat()
    log_message = f"[SECURITY] {event_type} - {timestamp} - {details}"
    
    if level == "INFO":
        logger.info(log_message)
    elif level == "WARNING":
        logger.warning(log_message)
    elif level == "ERROR":
        logger.error(log_message)
    elif level == "CRITICAL":
        logger.critical(log_message)


# Cost display utilities
def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def get_budget_status(current_cost: float, max_cost: float) -> Dict[str, Any]:
    """
    Get budget status information.
    
    Returns:
        Dictionary with budget status details
    """
    percentage = (current_cost / max_cost * 100) if max_cost > 0 else 0
    remaining = max(0, max_cost - current_cost)
    
    if percentage >= 100:
        status = "exceeded"
        alert_level = "critical"
    elif percentage >= 80:
        status = "warning"
        alert_level = "high"
    elif percentage >= 50:
        status = "caution"
        alert_level = "medium"
    else:
        status = "ok"
        alert_level = "low"
    
    return {
        'status': status,
        'alert_level': alert_level,
        'percentage': round(percentage, 1),
        'current': current_cost,
        'max': max_cost,
        'remaining': remaining
    }
