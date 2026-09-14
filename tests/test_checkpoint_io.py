import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from aw_ai.checkpoint_io import replace_with_retry, publish_report


class CheckpointIOTests(unittest.TestCase):
    def test_transient_lock_retries(self):
        with patch.object(Path, 'replace', side_effect=[PermissionError('locked'), None]) as replace:
            with patch('aw_ai.checkpoint_io.sleep') as sleep:
                replace_with_retry('new', 'old')
        self.assertEqual(replace.call_count, 2)
        sleep.assert_called_once_with(.1)

    def test_persistent_checkpoint_lock_raises(self):
        with patch.object(Path, 'replace', side_effect=PermissionError('locked')) as replace:
            with patch('aw_ai.checkpoint_io.sleep'):
                with self.assertRaises(PermissionError):replace_with_retry('new', 'old')
        self.assertEqual(replace.call_count, 6)

    def test_report_lock_preserves_old_and_recovers_next_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json';path.write_text('{"game": 1}')
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')):
                with patch('aw_ai.checkpoint_io.sleep'), self.assertWarns(RuntimeWarning):
                    self.assertFalse(publish_report(path, {'game':2}))
            self.assertEqual(path.read_text(), '{"game": 1}')
            self.assertIn('2',path.with_suffix('.json.tmp').read_text())
            self.assertTrue(publish_report(path, {'game':3}))
            self.assertIn('3',path.read_text())
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_other_io_error_is_not_swallowed(self):
        with patch.object(Path, 'replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):replace_with_retry('new','old')
