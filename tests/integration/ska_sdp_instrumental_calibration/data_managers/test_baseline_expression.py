import numpy as np
from mock import Mock

from ska_sdp_instrumental_calibration.data_managers import baseline_expression

BaselinesExpression = baseline_expression.BaselinesExpression


class TestBaselinesExpression:
    def test_init_negate_normalization_string(self):
        expr = BaselinesExpression(left="1", right="2", negate="!")
        assert expr.negate is True

    def test_init_negate_false(self):
        expr = BaselinesExpression(left="1", right="2", negate="")
        assert expr.negate is False

    def test_default_parser(self):
        expr = BaselinesExpression(left="1", right="2", negate=False)
        assert expr.antenna_parser("5") == 5

    def test_parse_range_and_single_value(self):
        expr = BaselinesExpression(left="5", right="6~8", negate=False)
        assert list(expr.left) == [5]
        assert list(expr.right) == [6, 7, 8]

        expr = BaselinesExpression(left="1~3", right="4", negate=False)
        assert list(expr.left) == [1, 2, 3]
        assert list(expr.right) == [4]

    def test_predicate_basic_match(self):
        expr = BaselinesExpression(left="1", right="2", negate=False)

        baselines = [(1, 2), (3, 4), (1, 5)]
        expected = np.array([True, False, False])

        result = expr.predicate(baselines)
        np.testing.assert_array_equal(result, expected)

    def test_predicate_range_match(self):
        expr = BaselinesExpression(left="1~2", right="3~4", negate=False)

        baselines = [(1, 3), (2, 4), (1, 5), (3, 3)]
        expected = np.array([True, True, False, False])

        result = expr.predicate(baselines)
        np.testing.assert_array_equal(result, expected)

    def test_predicate_negation(self):
        expr = BaselinesExpression(left="1", right="2", negate=True)

        baselines = [(1, 2), (3, 4)]
        expected = np.array([False, True])

        result = expr.predicate(baselines)
        np.testing.assert_array_equal(result, expected)

    def test_predicate_with_mock_parser(self):
        mock_parser = Mock(side_effect=lambda x: int(x.strip("A")))

        expr = BaselinesExpression(
            left="A1", right="A2~A3", negate=False, antenna_parser=mock_parser
        )

        baselines = [(1, 2), (1, 3), (1, 4)]

        result = expr.predicate(baselines)

        np.testing.assert_array_equal(result, np.array([True, True, False]))

        assert mock_parser.call_count >= 3
        mock_parser.assert_any_call("A1")
        mock_parser.assert_any_call("A2")
