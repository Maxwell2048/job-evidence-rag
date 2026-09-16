import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_public_release as release


class PublicReleaseTests(unittest.TestCase):
    def test_dangerous_and_nonempty_destinations_preserve_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            source = base / "source"
            source.mkdir()
            occupied = base / "occupied"
            occupied.mkdir()
            sentinel = occupied / "keep.txt"
            sentinel.write_text("keep")
            with patch.object(release, "PROJECT_ROOT", source):
                for target in (source, base, Path(base.anchor), source / "child", occupied):
                    with self.subTest(target=target), self.assertRaises(ValueError):
                        release.copy_tree(target)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_clean_flag_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            release.main(["--out", "unused", "--clean"])
        self.assertEqual(exc.exception.code, 2)

    def test_release_and_private_rules_without_copying_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            (source / "README.md").write_text("fictional example", encoding="utf-8")
            private = base / "private_patterns.json"
            private.write_text(json.dumps(["example-" + "private-marker"]))
            out = base / "release"
            with patch.multiple(release, PROJECT_ROOT=source, INCLUDE_FILES=["README.md"], INCLUDE_TESTS=[]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(release.main(["--out", str(out), "--private-patterns", str(private)]), 0)
                with self.assertRaises(ValueError):
                    release.copy_tree(out)
            self.assertTrue((out / "README.md").is_file())
            self.assertFalse((out / private.name).exists())

    def test_missing_source_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            out = Path(tmp) / "release"
            with patch.multiple(release, PROJECT_ROOT=source, INCLUDE_FILES=["missing.py"], INCLUDE_TESTS=[]), self.assertRaises(ValueError):
                release.copy_tree(out)
            self.assertFalse(out.exists())

    def test_script_and_license_are_scanned_without_exposing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = "ghp_" + "a" * 30
            for name in ("make_public_release.py", "LICENSE"):
                (root / name).write_text(token)
            hits = release.scan(root)
            self.assertEqual({h[0] for h in hits}, {"make_public_release.py", "LICENSE"})
            self.assertNotIn(token, repr(hits))

    def test_invalid_private_pattern_does_not_echo_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = Path(tmp) / "patterns.json"
            marker = "fictional-" + "private-value["
            private.write_text(json.dumps([marker]))
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = release.main(["--out", str(Path(tmp) / "out"), "--private-patterns", str(private)])
            self.assertEqual(code, 2)
            self.assertNotIn(marker, stderr.getvalue())
