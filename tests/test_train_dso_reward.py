"""Dependency-free regression tests for the DSO reward callback.

Ev is defined inside main(), so load its actual AST without importing the
training script or initializing TensorFlow, CUDA, Qlib, or market data.
The callback implementation is not copied into this test.
"""

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, call, patch


class OutOfDataRangeError(Exception):
    """Stand-in for the expression evaluator's exception in this isolated test."""


def load_evaluator(pool, expression, data_test, target):
    path = Path(__file__).resolve().parents[1] / "train_DSO.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    classes = [node for node in ast.walk(main)
               if isinstance(node, ast.ClassDef) and node.name == "Ev"]
    if len(classes) != 1:
        raise AssertionError("Expected exactly one Ev class inside main()")
    module = ast.parse("")
    module.body = classes
    namespace = {
        "OutOfDataRangeError": OutOfDataRangeError,
        "pool": pool,
        "test_expression": expression,
        "data_test": data_test,
        "target": target,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Ev"](pool)


class TestDSORewardCallback(unittest.TestCase):
    def setUp(self):
        self.pool = Mock()
        self.pool.test_ensemble.return_value = (0.08, 0.12)
        self.expression = object()
        self.data_test = object()
        self.target = object()
        self.ev = load_evaluator(
            self.pool, self.expression, self.data_test, self.target)

    def test_preserves_successful_pool_rewards(self):
        rewards = [-0.2, 0.0, 0.17, 0.83]
        self.pool.try_new_expr.side_effect = rewards
        with patch("builtins.print") as print_mock:
            for count, reward in enumerate(rewards, 1):
                with self.subTest(reward=reward):
                    self.assertEqual(
                        self.ev.alpha_ev_fn("test_expression"), reward)
                    self.assertEqual(self.ev.cnt, count)
            print_mock.assert_not_called()
        self.assertEqual(self.pool.try_new_expr.call_args_list,
                         [call(self.expression)] * len(rewards))
        self.pool.test_ensemble.assert_not_called()
        self.assertEqual(self.ev.results, {})

    def test_out_of_data_range_keeps_penalty(self):
        self.pool.try_new_expr.side_effect = OutOfDataRangeError
        self.assertEqual(self.ev.alpha_ev_fn("test_expression"), -1.0)
        self.assertEqual(self.ev.cnt, 1)
        self.pool.try_new_expr.assert_called_once_with(self.expression)
        self.pool.test_ensemble.assert_not_called()
        self.assertEqual(self.ev.results, {})

    def test_reporting_does_not_replace_successful_reward(self):
        self.ev.cnt = 99
        self.pool.try_new_expr.return_value = 0.42
        with patch("builtins.print") as print_mock:
            result = self.ev.alpha_ev_fn("test_expression")
        self.assertEqual(result, 0.42)
        self.assertEqual(self.ev.cnt, 100)
        self.assertEqual(self.ev.results, {100: 0.08})
        self.pool.try_new_expr.assert_called_once_with(self.expression)
        self.pool.test_ensemble.assert_called_once_with(
            self.data_test, self.target)
        print_mock.assert_called_once_with(100, 0.08)

    def test_out_of_data_range_still_reports_at_boundary(self):
        self.ev.cnt = 99
        self.pool.try_new_expr.side_effect = OutOfDataRangeError
        with patch("builtins.print") as print_mock:
            result = self.ev.alpha_ev_fn("test_expression")
        self.assertEqual(result, -1.0)
        self.assertEqual(self.ev.cnt, 100)
        self.assertEqual(self.ev.results, {100: 0.08})
        self.pool.test_ensemble.assert_called_once_with(
            self.data_test, self.target)
        print_mock.assert_called_once_with(100, 0.08)


if __name__ == "__main__":
    unittest.main()
