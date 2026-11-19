"""CPU-only checks for the six-GPU experiment's resource gate."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .native_suite import Suite


class ResourceGateTests(unittest.TestCase):
    def test_busy_original_gpu_prevents_container_launch(self):
        # Seven free GPUs still do not authorize changing the original mapping.
        status = "0, 40000, 99\n" + "".join(f"{i}, 0, 0\n" for i in range(1, 8))
        with tempfile.TemporaryDirectory() as directory:
            suite = Suite(Path(directory))
            with patch("subprocess.check_output", return_value=status), \
                    patch("time.sleep", side_effect=InterruptedError("stop test wait")), \
                    patch("subprocess.run") as run:
                with self.assertRaises(InterruptedError):
                    suite.start("text", 32, False)
                run.assert_not_called()
                self.assertFalse(suite.running_container)

    def test_requires_three_idle_samples_and_checks_ports(self):
        status = "".join(f"{i}, 0, 0\n" for i in range(8))
        with tempfile.TemporaryDirectory() as directory:
            suite = Suite(Path(directory))
            with patch("subprocess.check_output", return_value=status) as probe, \
                    patch("time.sleep") as sleep, patch("socket.socket") as socket:
                suite.wait_resources()
                self.assertEqual(probe.call_count, 3)
                self.assertEqual(sleep.call_count, 2)
                self.assertEqual(socket.return_value.__enter__.return_value.bind.call_count, 2)


if __name__ == "__main__":
    unittest.main()
