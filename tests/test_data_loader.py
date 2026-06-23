import pytest
import yaml
import pandas as pd
from src.data_loader import TextDataLoader

@pytest.fixture
def config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_load_data(config):
    loader = TextDataLoader(config)
    df = loader.load_data()
    assert len(df) == 7613
    assert 'text' in df.columns
    assert 'target' in df.columns

def test_split_data(config):
    loader = TextDataLoader(config)
    loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data()
    assert len(X_train) == int(7613 * 0.8)
    assert len(X_test) == 7613 - len(X_train)

def test_get_data(config):
    loader = TextDataLoader(config)
    X_train, X_test, y_train, y_test = loader.get_data()
    assert X_train is not None
    assert y_train is not None