"""Short notes the trainer prints between tqdm bar updates, without breaking the bar's line."""
import logging
import sys


def _is_console(h):
    return isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (sys.stdout, sys.stderr)


def _console_loggers(logger):
    """The loggers on `logger`'s way up to the root whose handlers print to the console."""
    out, lg = [], logger
    while lg is not None:
        if any(_is_console(h) for h in lg.handlers):
            out.append(lg)
        if not lg.propagate:
            break
        lg = lg.parent
    return out or [logging.root]


def say(logger, msg, level=logging.INFO):
    """Log a line through tqdm.write (tqdm's logging redirect), so it lands above the bar, not inside it - once.
    Only the loggers that really print to the console get redirected (normally just the root's basicConfig
    handler): redirecting `logger` as well gave it a console handler of its own, and the line came out twice,
    once bare and once again as INFO:__main__:... through the root."""
    try:
        from tqdm.contrib.logging import logging_redirect_tqdm
        with logging_redirect_tqdm(loggers=_console_loggers(logger)):
            logger.log(level, msg)
    except Exception:
        logger.log(level, msg)
