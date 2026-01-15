import unittest
from src.auth import parse_aws_credentials

class TestAuth(unittest.TestCase):
    def test_parse_aws_creds(self):
        text = """
        export AWS_ACCESS_KEY_ID="MYACCESSKEY"
        export AWS_SECRET_ACCESS_KEY="MYSECRETKEY"
        """
        creds = parse_aws_credentials(text)
        self.assertIsNotNone(creds)
        self.assertEqual(creds['access_key'], "MYACCESSKEY")
        self.assertEqual(creds['secret_key'], "MYSECRETKEY")
        
    def test_parse_aws_creds_invalid(self):
        text = "some random text"
        creds = parse_aws_credentials(text)
        self.assertIsNone(creds)

if __name__ == '__main__':
    unittest.main()
