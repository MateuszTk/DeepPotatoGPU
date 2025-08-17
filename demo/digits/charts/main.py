import matplotlib.pyplot as plt
import pandas as pd
import os

def load_time_data(directory, use_wmma, lowp_type, exec_type, threads=-1):
    end_of_filename = f"_0_{lowp_type}_{use_wmma}_{exec_type}_.txt"
    if threads >= 0:
        end_of_filename = f"_{threads}" + end_of_filename
    all_results = []
    for filename in os.listdir(directory):
        if filename.startswith("log_digits_") and filename.endswith(end_of_filename):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, sep=" ", header=None)
            df.columns = ["samplesTotal", "diff_ms", "forwardAvg", "backwardAvg", "updateAvg", "testAccuracy", "loss"]
            all_results.append(df)
    if not all_results:
        print("No results found in the directory. end_of_filename:", end_of_filename)
        return None
    combined_df = pd.concat(all_results)
    averaged_df = combined_df.groupby("samplesTotal").mean().reset_index()
    averaged_df = averaged_df.drop(columns=["testAccuracy", "loss"])
    return averaged_df

def load_accuracy_loss_data(directory, use_wmma, lowp_type, exec_type, threads=-1):
    end_of_filename = f"_1_{lowp_type}_{use_wmma}_{exec_type}_.txt"
    if threads >= 0:
        end_of_filename = f"_{threads}" + end_of_filename
    all_results = []
    for filename in os.listdir(directory):
        if filename.startswith("log_digits_") and filename.endswith(end_of_filename):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, sep=" ", header=None)
            df.columns = ["samplesTotal", "diff_ms", "forwardAvg", "backwardAvg", "updateAvg", "testAccuracy", "loss"]
            all_results.append(df)
    if not all_results:
        print("No accuracy/loss results found in the directory. end_of_filename:", end_of_filename)
        return None
    combined_df = pd.concat(all_results)
    averaged_df = combined_df.groupby("samplesTotal").mean().reset_index()
    return averaged_df

def load_and_average_results(directory, use_wmma, lowp_type, exec_type, threads=-1):
    averaged_time_df = load_time_data(directory, use_wmma, lowp_type, exec_type, threads)
    averaged_accuracy_df = load_accuracy_loss_data(directory, use_wmma, lowp_type, exec_type, threads)

    if averaged_time_df is None or averaged_accuracy_df is None:
        return None

    merged_df = averaged_time_df.merge(averaged_accuracy_df[["samplesTotal", "testAccuracy", "loss"]], on="samplesTotal", how="left")

    return merged_df

def load_threading_data():
    directory = "data/mt_test"
    threading_results = [(thread_cnt, load_time_data(directory, 0, "float", "class CPUExecutor", thread_cnt)) for thread_cnt in range(0, 20)]
    threading_results = [(thread_cnt, df) for thread_cnt, df in threading_results if df is not None and not df.empty]
    if not threading_results:
        print("No threading results found.")
        return None
    return threading_results
   


def plot_results(results_list, labels):
    if not results_list or all(r is None for r in results_list):
        print("No data to plot.")
        return

    # Time Metrics of Full Runs
    plt.figure(figsize=(12, 8))
    bar_width = 0.15
    groups = ["diff_ms", "forwardAvg", "backwardAvg", "updateAvg"]
    x = range(len(groups))
    for i, (df, label) in enumerate(zip(results_list, labels)):
        if df is not None:
            offset = i * bar_width
            plt.bar([x_val + offset for x_val in x], df[groups].mean().values, width=bar_width, label=label)
    plt.title("Time Metrics of Full Runs")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Test Accuracy Over Samples Total
    plt.figure(figsize=(12, 8))
    for df, label in zip(results_list, labels):
        if df is not None:
            plt.plot(df["samplesTotal"], df["testAccuracy"], label=f"Test Accuracy - {label}", marker='o')
    plt.title("Test Accuracy Over Samples Total")
    plt.xlabel("Samples")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Test Accuracy Over Time
    plt.figure(figsize=(12, 8))
    for df, label in zip(results_list, labels):
        if df is not None:
            plt.plot(df["diff_ms"], df["testAccuracy"], label=f"Test Accuracy - {label}", marker='o')
    plt.title("Test Accuracy Over Time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

def plot_threading_results(threading_results):
    plt.figure(figsize=(12, 8))
    x_values = [tr[0] for tr in threading_results]
    y_values = [tr[1]["diff_ms"].iloc[-1] / 1000 for tr in threading_results]
    plt.plot(x_values, y_values, label="Time (ms)", marker='o')
    plt.xlabel("Number of Threads")
    plt.ylabel("Time (s)")
    plt.title("Threading Results")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

directory = "data"
averaged_results_cpu_float = load_and_average_results(directory, use_wmma=0, lowp_type="float", exec_type="class CPUExecutor")
averaged_results_cuda_float = load_and_average_results(directory, use_wmma=0, lowp_type="float", exec_type="class CUDAExecutor")
averaged_results_cuda_half = load_and_average_results(directory, use_wmma=0, lowp_type="struct __half", exec_type="class CUDAExecutor")
averaged_results_wmma_half = load_and_average_results(directory, use_wmma=1, lowp_type="struct __half", exec_type="class CUDAExecutor")

results_list = [
    averaged_results_cpu_float,
    averaged_results_cuda_float,
    averaged_results_cuda_half,
    averaged_results_wmma_half
]
labels = [
    "CPU Float",
    "CUDA Float",
    "CUDA Half",
    "WMMA Half"
]

plot_results(results_list, labels)

threading_results = load_threading_data()
if threading_results:
    plot_threading_results(threading_results)



