import unittest

from evaluation.run_qwen3vl8b_all_maps import (
    canonical_qwen_model_id,
    qwen_model_matches,
    resolve_qwen_model_id,
)


class QwenRunnerModelIdentityTests(unittest.TestCase):
    def test_local_alias_matches_official_instruct_service_id(self):
        self.assertTrue(
            qwen_model_matches(
                "qwen3-vl-8b",
                "Qwen/Qwen3-VL-8B-Instruct",
            )
        )
        self.assertEqual(
            "qwen3vl8b",
            canonical_qwen_model_id("Qwen/Qwen3-VL-8B-Instruct"),
        )
        self.assertEqual(
            "Qwen/Qwen3-VL-8B-Instruct",
            resolve_qwen_model_id(
                "qwen3-vl-8b",
                ["Qwen/Qwen3-VL-8B-Instruct"],
            ),
        )

    def test_different_size_or_quantized_variant_does_not_match(self):
        self.assertFalse(
            qwen_model_matches(
                "qwen3-vl-8b",
                "Qwen/Qwen3-VL-32B-Instruct",
            )
        )
        self.assertFalse(
            qwen_model_matches(
                "qwen3-vl-8b",
                "Qwen/Qwen3-VL-8B-Instruct-AWQ",
            )
        )


if __name__ == "__main__":
    unittest.main()
