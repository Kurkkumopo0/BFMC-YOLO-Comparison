import argparse
import pandas as pd
from summarize import summarize
from ultralytics import YOLO


def train(model, data, epochs, name, batch):
    model = YOLO(model)
    results = model.train(
        data=data,
        epochs=epochs,
        name=name,
        batch=batch,
    )
    return results


def main(model, data, epochs, name, batch):
    try:
        results = train(model, data, epochs, name, batch)
        summarize(results, model=model)
    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLO training with different models.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Name of the yolo model.')
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path to the data.yml.')
    parser.add_argument('-e', '--epochs', type=int, default=40,
                        help='Number of training epochs. Default is 40.')
    parser.add_argument('-b', '--batch', type=int, default=-1,
                        help='Batch size used in training. Default is -1 (auto).')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='Name for the training run')
    args = parser.parse_args()

    main(args.model, args.data, args.epochs, args.name, args.batch)

