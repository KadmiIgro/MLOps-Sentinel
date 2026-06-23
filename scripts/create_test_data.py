import pandas as pd

def create_test_data():
    df = pd.DataFrame({
        'text': ['Fire in the building!', 'I love this weather', 'Earthquake detected', 'Just a normal day'],
        'target': [1, 0, 1, 0]
    })
    df.to_csv('data/train.csv', index=False)
    print("✅ data/train.csv создан (4 строки)")

if __name__ == "__main__":
    create_test_data()