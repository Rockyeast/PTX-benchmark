import unittest

from audit_direct_ptx_trajectory import audit_trajectory


class AuditDirectPtxTrajectoryTest(unittest.TestCase):
    def test_allows_regular_shell_tool_calls(self) -> None:
        data = {
            "steps": [
                {
                    "step_id": 1,
                    "tool_calls": [
                        {
                            "function_name": "exec",
                            "arguments": {"input": "ptxas /app/kernel.ptx -o /tmp/kernel.cubin"},
                        }
                    ],
                }
            ]
        }
        self.assertEqual(audit_trajectory(data), [])

    def test_rejects_wrapped_web_run(self) -> None:
        data = {
            "steps": [
                {
                    "step_id": 55,
                    "tool_calls": [
                        {
                            "function_name": "exec",
                            "arguments": {
                                "input": "const r = await tools.web__run({search_query:[{q:'PTX'}]});"
                            },
                        }
                    ],
                }
            ]
        }
        violations = audit_trajectory(data)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].step_id, "55")

    def test_rejects_direct_web_search_tool(self) -> None:
        data = {
            "steps": [
                {
                    "step_id": "search-1",
                    "tool_calls": [
                        {
                            "function_name": "web_search",
                            "arguments": {"query": "PTX ISA"},
                        }
                    ],
                }
            ]
        }
        self.assertEqual(len(audit_trajectory(data)), 1)

    def test_rejects_invalid_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "steps list"):
            audit_trajectory({"steps": None})


if __name__ == "__main__":
    unittest.main()
