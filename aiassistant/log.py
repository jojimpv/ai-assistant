"""
This script manages logging of the application.
The logging has console as well as file handlers.
The file handler logs messages into log file named “aiassistant.log”.
"""
import logging
import os
import time

log_level = getattr(logging, str(os.getenv('AIASSISTANT_LOG_LEVEL', 'INFO')).upper())
log_format = '%(asctime)s %(levelname)s %(name)s : %(message)s'
log_formatter = logging.Formatter(log_format)
logging.basicConfig(
    level=log_level,
    format=log_format,
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
root_logger = logging.getLogger()
root_handler = logging.FileHandler('../aiassistant.log')
root_handler.setLevel(log_level)
root_handler.setFormatter(log_formatter)
root_logger.addHandler(root_handler)
start_time = time.monotonic()
last_time = start_time

pdfminer = logging.getLogger('pdfminer')
pdfminer.setLevel(logging.INFO)


class CustomLogger(logging.LoggerAdapter):
    """
    Custom logger to add duration between previous and current log message
    """
    def process(self, msg, kwargs):
        time_str = time_check()['time']
        return '%s %s' % (msg, time_str), kwargs


def get_logger(name: str):
    """Get logger with given name.

    Args:
        name: Name of the logger

    Returns:
        Logger with given name

    """
    logger = logging.getLogger(name=name)
    logger_with_duration = CustomLogger(logger)
    return logger_with_duration


def time_check() -> dict[str, str]:
    """Get formatted time duration between last log message and current time.

    Returns:
        Dict with time duration
    """
    def convert_to_hh_mm_ss(seconds):
        mm, ss = divmod(seconds, 60)
        hh, mm = divmod(mm, 60)
        hms = "%d:%02d:%02d" % (hh, mm, ss)
        return hms
    global last_time
    curr_time = time.monotonic()
    diff_time = curr_time - start_time
    duration = curr_time - last_time
    last_time = curr_time
    result = dict(time=f'[{convert_to_hh_mm_ss(seconds=diff_time)}, {convert_to_hh_mm_ss(seconds=duration)}]')
    return result

