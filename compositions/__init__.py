"""Attack compositions on the existing generation/completion/evaluation CLIs."""


def generate(args):
    from .cli import generate as run
    return run(args)
