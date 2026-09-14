"""Atomic checkpoint publication resilient to temporary Windows file locks."""
import json
from pathlib import Path
from time import sleep
import warnings


def replace_with_retry(source, target):
    """Keep the old destination intact until replacement succeeds."""
    delays = (.1, .2, .4, .8, 1.6)
    for attempt in range(len(delays) + 1):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            sleep(delays[attempt])


def publish_report(path, report):
    """A report lock must not abort training after its checkpoint was saved.

    The checkpoint contains the same report in its metadata. Leave the .tmp
    report available for recovery and retry publication at the next save.
    """
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    try:
        temporary.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        replace_with_retry(temporary, path)
    except PermissionError as error:
        warnings.warn(f'Checkpoint saved, but progress report could not be updated: {path}. '
                      f'The report is preserved in checkpoint metadata. Will retry on the next save. {error}',
                      RuntimeWarning)
        return False
    return True
