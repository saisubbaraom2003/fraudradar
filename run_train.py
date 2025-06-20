import yaml
from src.data_loader import load_data
from src.preprocess import preprocess_features
from src.model import get_model
from src.train import train_and_evaluate

def main():
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)

    df = load_data(config['data_path'])
    X, y, scaler = preprocess_features(df)
    model = get_model(**config['model'])
    train_and_evaluate(X, y, model, config, scaler)

if __name__ == "__main__":
    main()
