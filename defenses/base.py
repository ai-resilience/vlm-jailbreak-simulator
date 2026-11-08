"""Common defense boundary around a loaded text/image target."""
import copy


class BaseDefense:
    name = ''

    def __init__(self, config=None):
        self.config = copy.deepcopy(config or {})
        self.config_path = None

    def preflight(self, text, images):
        """Validate one resolved input before creating a target model."""
        if not isinstance(text, str):
            raise ValueError('Defense input text must be a string')

    def validate_target(self, model_config):
        """Reject unsupported targets before loading model weights."""

    def prepare(self):
        """Load/check optional detector assets before target model creation."""

    def metadata(self):
        """Return serializable settings and implementation/resource provenance."""
        raise NotImplementedError

    def apply_input(self, text, images, target):
        """Return blocked, text/images, optional response/default_response, meta.

        A defense can query target internally. The caller generates once more
        only when no response is supplied and the input is allowed.
        """
        raise NotImplementedError
