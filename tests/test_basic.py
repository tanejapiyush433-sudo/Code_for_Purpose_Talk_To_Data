
import pandas as pd
from src.utils import preprocess

def test_preprocess():
    df = pd.DataFrame({'value': [1,2,3,4,5]})
    processed = preprocess(df)
    
    assert 'ds' in processed.columns
    assert 'y' in processed.columns
    assert len(processed) == 5
