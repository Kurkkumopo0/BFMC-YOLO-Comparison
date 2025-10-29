import platform
import psutil
import os
import math
import pandas as pd
import numpy as np

def _get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return gpu_name
    except:
        return "Not available"
    

def _get_system_info():
    cpu_info = f"CPU: {platform.processor()} ({psutil.cpu_count(logical=True)} cores)"
    ram_info = f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB total"
    gpu_info = f"GPU: {_get_gpu_info()}"
    return cpu_info, ram_info, gpu_info

def _get_total_time(df):
    time_steps = df['time'].diff().fillna(df['time'].iloc[0])
    total_time = 0
    for step in time_steps:
        if step < 0: # handle interrupts
            total_time += total_time + step
        else:
            total_time += step
    return total_time


def _get_avg_time(df):
    threshold_coefficent = 5
    time_steps = df['time'].diff().fillna(df['time'].iloc[0])
    valid_times = []
    for step in time_steps:
        if step < 0: # handle interrupts
            continue
        if not valid_times:
            valid_times.append(step)
            continue
        if np.mean(valid_times) < threshold_coefficent * step: # handle idle states
            continue
        valid_times.append(step)
    return np.mean(valid_times)


def _get_time_info(results_csv):
    df = pd.read_csv(results_csv)
    total_time = _get_total_time(df)
    time_steps = _get_avg_time(df)
    epochs = len(df)
    average_time = time_steps.mean()
    return epochs, total_time, average_time


def summarize(results, model="unknown"):
    run_dir = results.save_dir
    results_path = os.path.join(run_dir, "results.csv")
    summary_path = os.path.join(run_dir, "summary.txt")

    cpu, ram, gpu = _get_system_info()
    epochs, time, avg_time = _get_time_info(results_path)

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
