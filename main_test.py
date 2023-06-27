import unittest
from main import verify_files


class TestMain(unittest.TestCase):
    def test_verify(self):
        self.assertEqual(
            verify_files('name.csv','name.csv','name.csv'),True,
            "Should be csv files",
        )
    def test_verify_wrong(self):
        self.assertEqual(
            verify_files('name.mp3','name.csv','name.csv'),False,
            "Should be csv files",
        )
    def test_verify_wrong_case1(self):
        self.assertEqual(
            verify_files('name','name.csv','name.csv'),False,
            "Should be csv files",
        )


if __name__ == "__main__":
    unittest.main()
