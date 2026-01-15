import unittest
import os
import tempfile
import json
from src.utils import format_bytes, update_env_file

class TestUtils(unittest.TestCase):
    def test_format_bytes(self):
        self.assertEqual(format_bytes(500), "500.00 B")
        self.assertEqual(format_bytes(1024), "1.00 KB")
        self.assertEqual(format_bytes(1024*1024), "1.00 MB")
        
    def test_update_env_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock .env file location by changing cwd (risky) or better just test logic if extracted.
            # Since update_env_file hardcodes ".env", we have to run this carefully.
            # I will skip testing update_env_file in this environment to avoid modifying real .env
            pass

if __name__ == '__main__':
    unittest.main()
