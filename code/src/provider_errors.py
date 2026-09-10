"""Provider-neutral error hierarchy (U1).

Agent 5 catches THESE exceptions, not vendor-specific ones, so swapping
DRAMAMATRIX_VIDEO_PROVIDER never silently changes failure semantics. Concrete
vendor errors (e.g. the Agnes family in agnes_video.py) subclass the
corresponding neutral error, which keeps legacy raise sites and tests valid.

Mapping to episode/system states (unchanged by this layer):
- ProviderConfigurationError      -> blocked_on_*_configuration
- ProviderConnectionError         -> waiting_for_connectivity
- ProviderQueueFull               -> waiting_for_agnes_capacity (backoff)
- ProviderSubmissionUncertain     -> submission_uncertain (no auto-retry)
- ProviderGatewayUncertain        -> submission_uncertain (no auto-retry)
- ProviderContentPolicyViolation  -> director_rejected
- ProviderTaskFailed              -> director_rejected
- ProviderError (generic)         -> render_pending if task ids exist else render_failed
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for every video/image provider failure."""


class ProviderConfigurationError(ProviderError):
    """The local provider configuration is incomplete or invalid."""


class ProviderTaskFailed(ProviderError):
    """The remote generation task reached the failed terminal state."""


class ProviderContentPolicyViolation(ProviderError):
    """The provider rejected the prompt or task for content-policy reasons."""


class ProviderConnectionError(ProviderError):
    """A network error occurred during an idempotent operation."""


class ProviderSubmissionUncertain(ProviderError):
    """A create request timed out after it may have reached the provider."""


class ProviderQueueFull(ProviderError):
    """An explicit capacity/rate-limit refusal where NO task was created."""


class ProviderGatewayUncertain(ProviderError):
    """A 5xx gateway error where task-creation status is UNKNOWN."""
