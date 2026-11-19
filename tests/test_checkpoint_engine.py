"""End-to-end CPU checkpoint/restart test; no model download and no CUDA needed."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2

from colab.checkpoint_engine import restore_video, ffprobe_info


class FakeUpscaler:
    calls = 0
    fail_at = None
    tile = 64

    def enhance(self, frame):
        type(self).calls += 1
        if self.fail_at and type(self).calls >= self.fail_at:
            raise RuntimeError('simulated Colab disconnect')
        return cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg is required')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'input.mp4'
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'lavfi', '-i', 'testsrc2=size=48x32:rate=8:duration=3.25',
            '-f', 'lavfi', '-i', 'sine=frequency=480:duration=3.25',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest',
            str(self.source)
        ], check=True)
        self.weights = self.root / 'fake_model.pth'
        self.weights.write_bytes(b'model fixture')
        self.kwargs = dict(
            tile=64, segment_frames=10, crf=17, precision='fp32',
            stage_source=False, weights=self.weights,
            upscaler_factory=lambda *_: FakeUpscaler(), local_root=self.root / 'local')
        FakeUpscaler.calls, FakeUpscaler.fail_at = 0, None

    def run_job(self):
        return restore_video(self.source, self.root / 'output',
                             self.root / 'checkpoints', **self.kwargs)

    def test_resume_after_interruption_and_audio(self):
        FakeUpscaler.fail_at = 14
        with self.assertRaisesRegex(RuntimeError, 'simulated Colab disconnect'):
            self.run_job()
        ckpts = list((self.root / 'checkpoints').glob('*/checkpoint.json'))
        self.assertEqual(len(ckpts), 1)
        manifest = json.loads(ckpts[0].read_text())
        self.assertEqual(manifest['completed_frames'], 10)
        self.assertEqual(len(manifest['segments']), 1)
        FakeUpscaler.calls, FakeUpscaler.fail_at = 0, None
        result = self.run_job()
        self.assertEqual(FakeUpscaler.calls, 16)
        manifest = json.loads(ckpts[0].read_text())
        self.assertEqual(manifest['state'], 'done')
        self.assertEqual(manifest['completed_frames'], 26)
        info = ffprobe_info(result)
        self.assertEqual(info['streams'][0]['width'], 192)
        self.assertEqual(info['streams'][0]['height'], 128)
        self.assertTrue(any(s['codec_type'] == 'audio' for s in info['streams']))
        self.assertEqual(info['streams'][0]['codec_name'], 'h264')
        # Completed jobs must not re-run the model.
        FakeUpscaler.calls = 0
        self.assertEqual(self.run_job(), result)
        self.assertEqual(FakeUpscaler.calls, 0)

    def test_recover_orphan_segment(self):
        FakeUpscaler.fail_at = 12
        with self.assertRaisesRegex(RuntimeError, 'disconnect'):
            self.run_job()
        checkpoint = next((self.root / 'checkpoints').glob('*/checkpoint.json'))
        data = json.loads(checkpoint.read_text())
        # Simulate crash AFTER segment rename but BEFORE manifest update.
        data['segments'] = []
        data['completed_frames'] = 0
        checkpoint.write_text(json.dumps(data))
        FakeUpscaler.fail_at = None
        FakeUpscaler.calls = 0
        self.run_job()
        self.assertEqual(FakeUpscaler.calls, 16)


if __name__ == '__main__':
    unittest.main()
