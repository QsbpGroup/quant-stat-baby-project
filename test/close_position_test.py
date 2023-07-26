# 利用pytest框架编写测试用例
import pytest
import pandas as pd
from close_position import close_position


class TestClosePosition:

    def test_empty_df(self):
        df = pd.DataFrame(columns=['date', 'price'])
        with pytest.raises(ValueError, match='Input df must contain at least 2 rows.'):
            close_position(df)

    def test_not_dataframe(self):
        df = 'not a dataframe'
        with pytest.raises(TypeError, match='Input df must be a DataFrame.'):
            close_position(df)

    def test_not_float(self):
        df = pd.DataFrame(
            {'date': ['2022-01-01', '2022-01-02'], 'price': [100, 109]})
        threshold = 'not a float'
        with pytest.raises(TypeError, match='Input threshold must be a float.'):
            close_position(df, threshold)

    def test_negative_threshold(self):
        df = pd.DataFrame(
            {'date': ['2022-01-01', '2022-01-02'], 'price': [100, 109]})
        threshold = -0.1
        with pytest.raises(ValueError, match='Input threshold must be positive.'):
            close_position(df, threshold)

    def test_no_close_position(self):
        df = pd.DataFrame(
            {'date': ['2022-01-01', '2022-01-02'], 'price': [100, 109]})
        threshold = 0.1
        expected_output = pd.DataFrame(columns=['date', 'price', 'status'])
        assert close_position(df, threshold).equals(expected_output)

    def test_stop_loss(self):
        df = pd.DataFrame(
            {'date': ['2022-01-01', '2022-01-02', '2022-01-03'], 'price': [100, 90, 80]})
        threshold = 0.1
        expected_output = pd.DataFrame(
            {'date': ['2022-01-02'], 'price': [90], 'status': [0]})
        assert close_position(df, threshold).equals(expected_output)

    def test_stop_profit(self):
        df = pd.DataFrame(
            {'date': ['2022-01-01', '2022-01-02', '2022-01-03'], 'price': [100, 110, 120]})
        threshold = 0.1
        expected_output = pd.DataFrame(
            {'date': ['2022-01-02'], 'price': [110], 'status': [1]})
        assert close_position(df, threshold).equals(expected_output)
