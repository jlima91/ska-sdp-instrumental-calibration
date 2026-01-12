import numpy as np
from mock import Mock

from ska_sdp_instrumental_calibration.data_managers import baseline_expression

BaselinesExpression = baseline_expression.BaselinesExpression


class TestBaselinesExpression:
    def test_default_parser(self):
        expr = BaselinesExpression(left="1", right="2")
        assert expr.antenna_parser("5") == 5

    def test_parse_range_and_single_value(self):
        expr = BaselinesExpression(left="5", right="6~8")
        assert list(expr.left) == [5]
        assert list(expr.right) == [6, 7, 8]

        expr = BaselinesExpression(left="1~3", right="4")
        assert list(expr.left) == [1, 2, 3]
        assert list(expr.right) == [4]

    def test_predicate_basic_match(self):
        expr = BaselinesExpression(left="1", right="2")

        baselines = [(1, 2), (3, 4), (1, 5)]
        expected = np.array([False, True, True])

        result = expr.predicate(baselines)
        np.testing.assert_array_equal(result, expected)

    def test_predicate_range_match(self):
        expr = BaselinesExpression(left="1~2", right="3~4")

        baselines = [(1, 3), (1, 5), (2, 3), (2, 4), (3, 3)]
        expected = np.array([False, True, False, False, True])

        result = expr.predicate(baselines)
        np.testing.assert_array_equal(result, expected)

    def test_predicate_negation(self):
        expr = BaselinesExpression(left="1", right="2")

        baselines = [(1, 2), (3, 4)]
        expected = np.array([False, True])

        result = expr.predicate(baselines)
        np.testing.assert_array_equal(result, expected)

    def test_predicate_with_mock_parser(self):
        mock_parser = Mock(side_effect=lambda x: int(x.strip("A")))

        expr = BaselinesExpression(
            left="A1", right="A2~A3", antenna_parser=mock_parser
        )

        baselines = [(1, 2), (1, 3), (1, 4)]

        result = expr.predicate(baselines)

        np.testing.assert_array_equal(result, np.array([False, False, True]))

        assert mock_parser.call_count >= 3
        mock_parser.assert_any_call("A1")
        mock_parser.assert_any_call("A2")
