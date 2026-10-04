import unittest
from src.utils.config import load_config, PROJECT_ROOT

class TestConfiguration(unittest.TestCase):
    def test_project_root_resolution(self):
        self.assertEqual(PROJECT_ROOT.name, "dl-stock-prediction")

    def test_load_settings(self):
        config = load_config("settings.yaml")
        self.assertIn("project", config)
        self.assertIn("tickers", config["project"])
        self.assertEqual(config["features"]["sequence_length"], 30)

if __name__ == '__main__':
    unittest.main()
