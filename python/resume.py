import argparse
import os
from summarize import summarize
from ultralytics import YOLO


def resume(path):
    model = YOLO(path)
    results = model.train(resume=True)
    return results, model.model_name


def main(path):
    try:
        results, model_name = resume(path)
        summarize(results, model=model_name)
    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume training in case of interruption.')
    parser.add_argument('-p', '--path', type=str,
                        help='Path to the latest model top be resumed.')
    parser.add_argument('-n', '--name', type=str,
                        help='Name of the run to be resumed.')
    args = parser.parse_args()

    path = getattr(args, 'path') 
    name = getattr(args, 'name')  
    
    if not path and not name:
        parser.error("Path or name must be passed.")
    
    path_to_last = path if path else os.path.join('runs', 'detect', name, 'weights', 'last.pt')
    main(path_to_last)