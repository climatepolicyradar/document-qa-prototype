import unittest
from unittest.mock import patch, MagicMock
from src.controllers.LibraryManager import LibraryManager

class TestLibraryManager(unittest.TestCase):

    @patch('src.controllers.LibraryManager.get_secret')
    @patch('requests.get')
    def setUp(self, mock_get, mock_get_secret):
        mock_get_secret.return_value = "fake_token"
        self.library_manager = LibraryManager()
        self.mock_get = mock_get

    def test_get_metadata_for_citation(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test_metadata"}
        self.mock_get.return_value = mock_response

        result = self.library_manager.get_metadata_for_citation("doc_id", [1, 2, 3])
        self.mock_get.assert_called_once()
        self.assertEqual(result, {"data": "test_metadata"})

    def test_get_text_blocks_around(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test_text_blocks"}
        self.mock_get.return_value = mock_response

        result = self.library_manager.get_text_blocks_around("doc_id", 25, 10)
        self.mock_get.assert_called_once()
        self.assertEqual(result, {"data": "test_text_blocks"})

    def test_get_full_text(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test_full_text"}
        self.mock_get.return_value = mock_response

        result = self.library_manager.get_full_text("doc_id")
        self.mock_get.assert_called_once()
        self.assertEqual(result, {"data": "test_full_text"})

    def test_get_section_containing_block(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test_section"}
        self.mock_get.return_value = mock_response

        result = self.library_manager.get_section_containing_block("doc_id", 25)
        self.mock_get.assert_called_once()
        self.assertEqual(result, {"data": "test_section"})

if __name__ == '__main__':
    unittest.main()
