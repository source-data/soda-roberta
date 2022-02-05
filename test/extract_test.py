import unittest
from pathlib import Path
from smtag.extract import ExtractorXML
from smtag.config import config


class TestExtractorXML(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.content = [
            '<xml><b>This was it. Maybe it is not.</b></xml>',
            '<xml><b>This <g>is</g> not.</b> It!</xml>'
        ]
        cls.expected_examples = ['This was it.', 'Maybe it is not.', 'This is not.']
        cls.dest_dir = Path("/data/text/testing")
        cls.dest_dir.mkdir(exist_ok=True)
        cls.dest_filepath = cls.dest_dir / "dest_file.text"
        cls.source_dir = Path('/data/xml/tmp')
        cls.source_dir.mkdir(exist_ok=True)
        for i, e in enumerate(cls.content):
            p = cls.source_dir / f"test_file_{str(i)}.xml"
            p.write_text(e)
        xtract = ExtractorXML(source_dir=cls.source_dir)
        xtract.run(cls.dest_filepath, selector='.//b', punkt=True, keep_xml=False, min_length=5)
        cls.created_filenames = [f.name for f in cls.dest_dir.iterdir()]
        print("created files:", cls.created_filenames)

    def test_extracted_content(self):
        with self.dest_filepath.open() as f:
            examples = f.readlines()
            examples = [e.strip() for e in examples]
            examples = set(examples)
            expected = set(self.expected_examples)
            self.assertSetEqual(examples, expected)

    @classmethod
    def tearDownClass(cls):
        for f in cls.source_dir.glob("*.*"):
            f.unlink()
        for f in cls.dest_dir.glob("*.*"):
            f.unlink()
        cls.source_dir.rmdir()
        cls.dest_dir.rmdir()
        print("Cleaned up and removed testing dir and files")

if __name__ == '__main__':
    unittest.main()
