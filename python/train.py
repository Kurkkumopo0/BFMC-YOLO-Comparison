import argparse
import platform
import psutil
import os
import pandas as pd
import math
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


def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return gpu_name
    except:
        return "Not available"
    

def get_system_info():
    cpu_info = f"CPU: {platform.processor()} ({psutil.cpu_count(logical=True)} cores)"
    ram_info = f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB total"
    gpu_info = f"GPU: {get_gpu_info()}"
    return cpu_info, ram_info, gpu_info


def get_time_info(results_csv):
    df = pd.read_csv(results_csv)
    total_time = df['time'].iloc[-1]
    time_steps = df['time'].diff().fillna(df['time'].iloc[0])
    average_time = time_steps.mean()
    return total_time, average_time


def summarize(model, epochs, results):
    run_dir = results.save_dir
    results_path = os.path.join(run_dir, "results.csv")
    summary_path = os.path.join(run_dir, "summary.txt")

    cpu, ram, gpu = get_system_info()
    time, avg_time = get_time_info(results_path)

    hours = math.floor(time / (60 * 60))
    minutes = math.floor(time % (60 * 60) / 60)
    seconds = math.floor(time % (60 * 60) % 60)
    avg_minutes = math.floor(avg_time / 60)
    avg_seconds = math.floor(avg_time % 60)

    time_string = f"{hours}h, {minutes}m, {seconds}s"
    avg_string = f"{avg_minutes}m, {avg_seconds}s"

    with open(summary_path, 'w') as f:
        f.write(f"{model} TRAINING SUMMARY\n\n")

        f.write("SYSTEM SPECS\n")
        f.write(f"{cpu}\n")
        f.write(f"{ram}\n")
        f.write(f"{gpu}\n\n")

        f.write("DETAILS\n")
        f.write(f"Total Epochs Completed: {epochs}\n")
        f.write(f"Total Training Time: {time_string}\n")
        f.write(f"Average Time Per Epoch: {avg_string}\n\n")

        f.write("INFERENCE SPEED\n")
        for key, value in results.speed.items():
            f.write(f"{key}: {value:.4f}ms\n")
        f.write("\n")
        
        f.write("RESULTS\n")
        f.write(f"mAP95: {results.box.map:.3f}\n")
        f.write(f"mAP50: {results.box.map50:.3f}\n")
        f.write(f"Precision: {results.box.mp:.3f}\n")
        f.write(f"Recall: {results.box.mr:.3f}\n")
        f.write(f"Fitness: {results.fitness:.3f}\n\n")

        f.write("CLASS-WISE PERFORMANCE\n")
        f.write(f"{'Class':<10} {'Name':<20} {'Instances':<10} {'P':<10} {'R':<10} {'mAP50':<10} {'mAP95':<10}\n")
        for i, class_idx in enumerate(results.ap_class_index):
            class_name = results.names[class_idx]
            instances = results.nt_per_class[i]
            p = results.box.p[i]
            r = results.box.r[i]
            mAP50 = results.box.ap50[i]
            mAP95 = results.box.ap[i]
            f.write(f"{class_idx:<10} {class_name:<20} {instances:<10} {p:<10.3f} {r:<10.3f} {mAP50:<10.3f} {mAP95:<10.3f}\n")


def main(model, data, epochs, name, batch):
    try:
        results = train(model, data, epochs, name, batch)
        summarize(model, epochs, results)
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

