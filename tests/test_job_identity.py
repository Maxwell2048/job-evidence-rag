import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import job_identity

JD = ("Highlights\nBuild practical AI products.\n\nGraduate AI Engineer\n\nAbout the company\n"
      "Acme Analytics is part of Example Holdings Limited, an Australian technology company.\n")


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.errors_seen = []

    def generate_json(self, task, system_prompt, payload, schema=None, timeout_seconds=None, validate=None):
        for response in self.responses:
            errors = validate(response)
            self.errors_seen.append(errors)
            if not errors:
                return response
        raise job_identity.LLMError("no valid response", error_type="structure")


class IdentityTests(unittest.TestCase):
    def test_values_must_be_words_of_the_jd(self):
        self.assertEqual(job_identity.validate_identity(
            {"company": "acme  analytics", "job_title": "Graduate AI Engineer"}, JD), [])
        self.assertEqual(job_identity.validate_identity({"company": None, "job_title": None}, JD), [])
        errors = job_identity.validate_identity({"company": "Globex", "job_title": " "}, JD)
        self.assertEqual(len(errors), 2)
        self.assertIn("Globex", errors[0])

    def test_invented_company_is_rejected_and_retried(self):
        llm = FakeLLM([{"company": "Globex", "job_title": "Graduate AI Engineer"},
                       {"company": "Acme Analytics", "job_title": "Graduate AI Engineer"}])
        identity = job_identity.identify(llm, JD)
        self.assertEqual((identity["company"], identity["job_title"]), ("Acme Analytics", "Graduate AI Engineer"))
        self.assertTrue(llm.errors_seen[0])

    def test_load_identity_tolerates_missing_or_broken_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(job_identity.load_identity(tmp), {})
            (Path(tmp) / job_identity.IDENTITY_FILE).write_text("not json", encoding="utf-8")
            self.assertEqual(job_identity.load_identity(tmp), {})
            (Path(tmp) / job_identity.IDENTITY_FILE).write_text(json.dumps({"company": "Acme"}), encoding="utf-8")
            self.assertEqual(job_identity.load_identity(tmp), {"company": "Acme"})


if __name__ == "__main__":
    unittest.main()
