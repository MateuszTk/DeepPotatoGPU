import matplotlib.pyplot as plt
import pandas as pd
import os

def parse_filename(filename):
    #0   1      2 3   4 5 6  7  8 9 10            11 12 
    #log_digits_0_784_4_4_10_32_6_0_struct __half_1_class CUDAExecutor_
    parts = filename.split('_')
    #print(parts)
    if len(parts) < 13:
        return None
    log = parts[0]
    digits = parts[1]
    test_iter = int(parts[2])
    input_layer = int(parts[3])
    hidden_layer_1 = int(parts[4])
    hidden_layer_2 = int(parts[5])
    output_layer = int(parts[6])
    train_batch = int(parts[7])
    cpu_workers = int(parts[8])
    test_set = int(parts[9])
    lowp_type = parts[10]
    index_offset = 0
    if lowp_type == "struct ":
        lowp_type = "struct __half"
        index_offset = 2
    # Check if the lowp_type is 'struct __half' or 'float'
    use_wmma = int(parts[11 + index_offset])
    exec_type = parts[12 + index_offset]

    return {
        "log": log,
        "digits": digits,
        "test_iter": test_iter,
        "input_layer": input_layer,
        "hidden_layer_1": hidden_layer_1,
        "hidden_layer_2": hidden_layer_2,
        "output_layer": output_layer,
        "train_batch": train_batch,
        "cpu_workers": cpu_workers,
        "test_set": test_set,
        "lowp_type": lowp_type,
        "use_wmma": use_wmma,
        "exec_type": exec_type
    }

def load_time_data(directory, use_wmma, lowp_type, exec_type, threads=-1, hidden_layer_1=-1):
    end_of_filename = f"_0_{lowp_type}_{use_wmma}_{exec_type}_.txt"
    if threads >= 0:
        end_of_filename = f"_{threads}" + end_of_filename
    all_results = []
    for filename in os.listdir(directory):
        properties = parse_filename(filename)
        if filename.startswith("log_digits_") and filename.endswith(end_of_filename) and (properties["hidden_layer_1"] == hidden_layer_1 or hidden_layer_1 == -1):
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

def load_scaling_data(directory, use_wmma, lowp_type, exec_type, hidden_layers):
    all_results = []
    for hidden_layer_1 in hidden_layers:
        df = load_time_data(directory, use_wmma, lowp_type, exec_type, hidden_layer_1=hidden_layer_1)
        if df is not None:
            all_results.append(df.iloc[-1:])
    if not all_results:
        print("No scaling results found.")
        return None
    return {
        "hidden_layers": hidden_layers,
        "use_wmma": use_wmma,
        "lowp_type": lowp_type,
        "exec_type": exec_type,
        "data": all_results
    }
   

def plot_results(results_list, labels):
    if not results_list or all(r is None for r in results_list):
        print("No data to plot.")
        return

    # Time Metrics of Full Runs
    plt.figure(figsize=(12, 8))
    bar_width = 0.15
    groups = ["diff_ms", "forwardAvg", "backwardAvg", "updateAvg"]
    group_labels = ["Total Time", "Forward Pass", "Backward Pass", "Update"]
    x = range(len(groups))
    for i, (df, label) in enumerate(zip(results_list, labels)):
        if df is not None:
            offset = i * bar_width
            plt.bar([x_val + offset for x_val in x], df[groups].mean().values / 1000.0, width=bar_width, label=label)
    plt.title("Time Metrics of Full Runs")
    plt.ylabel("Time (s)")
    plt.xticks([r + bar_width * (len(results_list) - 1) / 2 for r in x], group_labels)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_metrics_full_runs.png")

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
    plt.savefig("test_accuracy_over_samples.png")

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
    plt.savefig("test_accuracy_over_time.png")

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
    plt.savefig("threading_results.png")
    plt.show()

def plot_scaling_data(all_results_total):
    plt.figure(figsize=(12, 8))
    #print(all_results_total)
    for result in all_results_total:
        if result is not None:
            hidden_layers = result["hidden_layers"]
            use_wmma = result["use_wmma"]
            lowp_type = result["lowp_type"]
            exec_type = result["exec_type"]
            df = result["data"]
            label = f"{exec_type}, {'float' if lowp_type == 'float' else 'half'}, {'WMMA' if use_wmma else ''}"
            plt.plot(hidden_layers, [d["diff_ms"].iloc[-1] / 1000 for d in df], label=label, marker='o')
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Time (s)")
    plt.title("Scaling Results")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scaling_results.png")

    # the same plot but with log scale
    plt.figure(figsize=(12, 8))
    for result in all_results_total:
        if result is not None:
            hidden_layers = result["hidden_layers"]
            use_wmma = result["use_wmma"]
            lowp_type = result["lowp_type"]
            exec_type = result["exec_type"]
            df = result["data"]
            label = f"{exec_type}, {'float' if lowp_type == 'float' else 'half'}, {'WMMA' if use_wmma else ''}"
            plt.plot(hidden_layers, [d["diff_ms"].iloc[-1] / 1000 for d in df], label=label, marker='o')
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Time (s)")
    plt.title("Scaling Results (Log Scale)")
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scaling_results_log.png")

    # plot of speedup factor compared to CPU float
    plt.figure(figsize=(12, 8))
    cpu_data = None
    for result in all_results_total:
        if result is not None:
            hidden_layers = result["hidden_layers"]
            use_wmma = result["use_wmma"]
            lowp_type = result["lowp_type"]
            exec_type = result["exec_type"]
            df = result["data"]
            if exec_type == "class CPUExecutor":
                cpu_data = df
                break
    
    max_speedups = {}
    min_speedups = {}
    if cpu_data is not None:
        cpu_times = [d["diff_ms"].iloc[-1] for d in cpu_data]
        for result in all_results_total:
            if result is not None:
                hidden_layers = result["hidden_layers"]
                use_wmma = result["use_wmma"]
                lowp_type = result["lowp_type"]
                exec_type = result["exec_type"]
                df = result["data"]
                label = f"{exec_type}, {'float' if lowp_type == 'float' else 'half'}, {'WMMA' if use_wmma else ''}"
                times = [d["diff_ms"].iloc[-1] for d in df]
                speedup = [cpu / time if time > 0 else 0 for cpu, time in zip(cpu_times, times)]
                plt.plot(hidden_layers, speedup, label=label, marker='o')
                max_speedups[label] = {
                    "max_speedup": max(speedup),
                    "layer_size": hidden_layers[speedup.index(max(speedup))]
                }
                min_speedups[label] = {
                    "min_speedup": min(speedup),
                    "layer_size": hidden_layers[speedup.index(min(speedup))]
                }

    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Speedup Factor")
    plt.legend()
    plt.title("Speedup Factor Compared to CPU Float")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scaling_results_speedup.png")
    plt.show()

    # Print max speedup information as a table
    print("Max Speedup Information:")
    print(f"{'Configuration':<50} {'Max Speedup':<15} {'Layer Size':<15}")
    for label, info in max_speedups.items():
        print(f"{label:<50} {info['max_speedup']:<15.2f} {info['layer_size']:<15}")

    # Print min speedup information as a table
    print("Minimum Speedup Information:")
    print(f"{'Configuration':<50} {'Min Speedup':<15} {'Layer Size':<15}")
    for label, info in min_speedups.items():
        print(f"{label:<50} {info['min_speedup']:<15.2f} {info['layer_size']:<15}")

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


scaling_directory = "data/scale_test"
hidden_layer_sizes = [ 4, 6, 8, 10, 12, 14, 16, 18, 20, 28, 32, 64, 128, 256, 512, 768, 1024 ]
scaling_data_cuda_float = load_scaling_data(scaling_directory, use_wmma=0, lowp_type="float", exec_type="class CUDAExecutor", hidden_layers=hidden_layer_sizes)
scaling_data_cuda_wmma_half = load_scaling_data(scaling_directory, use_wmma=1, lowp_type="struct __half", exec_type="class CUDAExecutor", hidden_layers=hidden_layer_sizes)
scaling_data_cpu_float = load_scaling_data(scaling_directory, use_wmma=0, lowp_type="float", exec_type="class CPUExecutor", hidden_layers=hidden_layer_sizes)
all_results_total = [scaling_data_cuda_float, scaling_data_cuda_wmma_half, scaling_data_cpu_float]
plot_scaling_data(all_results_total)
