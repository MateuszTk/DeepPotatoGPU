import matplotlib.pyplot as plt
import pandas as pd
import os

def executor_label(exec_type, data_type, use_wmma):
    label = exec_type.replace("class ", "").replace("Executor", "").strip()
    if data_type == "float":
        label += ", Float"
    else:
        label += ", Half"
    if use_wmma:
        label += ", WMMA"
    return label

def get_label_color(label):
    if "CPU" in label:
        return "blue"
    elif "CUDA" in label and "Float" in label:
        return "orange"
    elif "CUDA" in label and "Half" in label and "WMMA" not in label:
        return "green"
    elif "CUDA" in label and "Half" in label and "WMMA" in label:
        return "red"
    else:
        return "black"
    
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
    end_of_filename = f"_0_{lowp_type}_{use_wmma}_{exec_type}_.csv"
    if threads >= 0:
        end_of_filename = f"_{threads}" + end_of_filename
    all_results = []
    for filename in os.listdir(directory):
        properties = parse_filename(filename)
        if filename.startswith("log_digits_") and filename.endswith(end_of_filename) and (properties["hidden_layer_1"] == hidden_layer_1 or hidden_layer_1 == -1):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, sep=",", header=None)
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
    end_of_filename = f"_1_{lowp_type}_{use_wmma}_{exec_type}_.csv"
    if threads >= 0:
        end_of_filename = f"_{threads}" + end_of_filename
    all_results = []
    for filename in os.listdir(directory):
        if filename.startswith("log_digits_") and filename.endswith(end_of_filename):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, sep=",", header=None)
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
    print("Loading threading data...")
    directory = "logs/threading"
    threading_results = [(thread_cnt, load_time_data(directory, 0, "float", "class CPUExecutor", thread_cnt)) for thread_cnt in range(0, 33)]
    threading_results = [(thread_cnt, df) for thread_cnt, df in threading_results if df is not None and not df.empty]
    if not threading_results:
        print("No threading results found.")
        return None
    return threading_results

def load_scaling_data(directory, use_wmma, lowp_type, exec_type, hidden_layers):
    print(f"Loading scaling data for use_wmma={use_wmma}, lowp_type={lowp_type}, exec_type={exec_type}")
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

def load_power_data(directory, use_wmma, lowp_type, exec_type, hidden_layer_1):
    end_of_filename = f"_energy.csv"
    all_results = []
    for filename in os.listdir(directory):
        properties = parse_filename(filename)
        if filename.startswith("log_digits_") and filename.endswith(end_of_filename) and properties["use_wmma"] == use_wmma and properties["lowp_type"] == lowp_type and properties["exec_type"] == exec_type and (properties["hidden_layer_1"] == hidden_layer_1 or hidden_layer_1 == -1):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, sep=",")
            total_energy = df[df['Component'] == 'Total']['Energy (Wh)'].values[0]
            total_time = df[df['Component'] == 'Total']['Time (s)'].values[0]
            average_power = df[df['Component'] == 'Total']['Average Power (W)'].values[0]
            cpu_energy = df[df['Component'] == 'CPU']['Energy (Wh)'].values[0]
            gpu_energy = df[df['Component'] == 'GPU']['Energy (Wh)'].values[0]
            cpu_power = df[df['Component'] == 'CPU']['Average Power (W)'].values[0]
            gpu_power = df[df['Component'] == 'GPU']['Average Power (W)'].values[0]
            all_results.append({
                "total_energy_Wh": total_energy,
                "total_time_s": total_time,
                "average_power_W": average_power,
                "cpu_energy_Wh": cpu_energy,
                "gpu_energy_Wh": gpu_energy,
                "cpu_power_W": cpu_power,
                "gpu_power_W": gpu_power
            })
    if not all_results:
        print("No power results found in the directory. end_of_filename:", end_of_filename)
        return None
    combined_df = pd.DataFrame(all_results)
    averaged_df = combined_df.mean().to_dict()
    return averaged_df

def load_all_power_data(directory, use_wmma, lowp_type, exec_type, hidden_layers):
    print(f"Loading power data for use_wmma={use_wmma}, lowp_type={lowp_type}, exec_type={exec_type}")
    all_results = []
    for hidden_layer_1 in hidden_layers:
        power_data = load_power_data(directory, use_wmma, lowp_type, exec_type, hidden_layer_1)
        if power_data is not None:
            all_results.append({
                "hidden_layer_1": hidden_layer_1,
                "power_data": power_data
            })
    if not all_results:
        print("No power results found.")
        return None
    return {
        "use_wmma": use_wmma,
        "lowp_type": lowp_type,
        "exec_type": exec_type,
        "hidden_layers": hidden_layers,
        "power_data": all_results
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
            plt.bar([x_val + offset for x_val in x], df[groups].mean().values / 1000.0, width=bar_width, label=label, color=get_label_color(label))
    plt.title("Training time")
    plt.ylabel("Time (s)")
    plt.xticks([r + bar_width * (len(results_list) - 1) / 2 for r in x], group_labels)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_metrics_full_runs.png")

    # Print Time Metrics Table
    print("Time Metrics of Full Runs (s):")
    print(f"{'Configuration':<30} {'Total Time':<15} {'Forward Pass':<15} {'Backward Pass':<15} {'Update':<15}")
    for df, label in zip(results_list, labels):
        if df is not None:
            total_time = df["diff_ms"].mean() / 1000.0
            forward_time = df["forwardAvg"].mean() / 1000.0
            backward_time = df["backwardAvg"].mean() / 1000.0
            update_time = df["updateAvg"].mean() / 1000.0
            print(f"{label:<30} {total_time:<15.2f} {forward_time:<15.2f} {backward_time:<15.2f} {update_time:<15.2f}")

    # Print Time Metrics as Speedup Multipliers
    print("\nTime Metrics as Speedup Multipliers of CPU Float:")
    cpu_float_df = results_list[0]  # Assuming first is CPU Float
    for df, label in zip(results_list, labels):
        if df is not None and cpu_float_df is not None:
            total_time_ratio = (cpu_float_df["diff_ms"].mean() / df["diff_ms"].mean())
            forward_time_ratio = (cpu_float_df["forwardAvg"].mean() / df["forwardAvg"].mean())
            backward_time_ratio = (cpu_float_df["backwardAvg"].mean() / df["backwardAvg"].mean())
            update_time_ratio = (cpu_float_df["updateAvg"].mean() / df["updateAvg"].mean())
            print(f"{label:<30} {total_time_ratio:<15.2f} {forward_time_ratio:<15.2f} {backward_time_ratio:<15.2f} {update_time_ratio:<15.2f}")

    # Test Accuracy Over Samples Total
    plt.figure(figsize=(12, 8))
    for df, label in zip(results_list, labels):
        if df is not None:
            plt.plot(df["samplesTotal"], df["testAccuracy"], label=label, color=get_label_color(label))
    plt.title("Test accuracy over samples trained")
    plt.xlabel("Samples")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_accuracy_over_samples.png")

    # Test Accuracy Over Time
    plt.figure(figsize=(12, 8))
    for df, label in zip(results_list, labels):
        if df is not None:
            plt.plot(df["diff_ms"], df["testAccuracy"], label=label, color=get_label_color(label))
    plt.title("Test accuracy over time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Test accuracy")
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
    plt.title("Training time with multithreading")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("threading_results.png")

    # Print Threading Results Table With Speedup
    plt.figure(figsize=(12, 8))
    speedups = []
    print("Threading Results:")
    print(f"{'Threads':<10} {'Time (s)':<15} {'Speedup':<15}")
    for i, tr in enumerate(threading_results):
        if tr is not None:
            time = tr[1]["diff_ms"].iloc[-1] / 1000
            speedup = (threading_results[0][1]["diff_ms"].iloc[-1] / 1000) / time
            print(f"{i:<10} {time:<15.2f} {speedup:<15.2f}")
            speedups.append(speedup)

    plt.plot(x_values, speedups, label="Speedup", marker='o', color='orange')
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup factor")
    plt.title("Multithreading speedup factor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("threading_speedup.png")
    plt.show()

def plot_scaling_data(all_results_total):
    plt.figure(figsize=(12, 8))
    for result in all_results_total:
        if result is not None:
            hidden_layers = result["hidden_layers"]
            use_wmma = result["use_wmma"]
            lowp_type = result["lowp_type"]
            exec_type = result["exec_type"]
            df = result["data"]
            label = executor_label(exec_type, lowp_type, use_wmma)
            plt.plot(hidden_layers, [d["diff_ms"].iloc[-1] / 1000 for d in df], label=label, marker='o', color=get_label_color(label))
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Time (s)")
    plt.title("Training time depending on the hidden layer size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scaling_results.png")

    # Log scale plot
    plt.figure(figsize=(12, 8))
    for result in all_results_total:
        if result is not None:
            hidden_layers = result["hidden_layers"]
            use_wmma = result["use_wmma"]
            lowp_type = result["lowp_type"]
            exec_type = result["exec_type"]
            df = result["data"]
            label = executor_label(exec_type, lowp_type, use_wmma)
            plt.plot(hidden_layers, [d["diff_ms"].iloc[-1] / 1000 for d in df], label=label, marker='o', color=get_label_color(label))
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Time (s)")
    plt.title("Scaling Results (Log Scale)")
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scaling_results_log.png")

    # Speedup factor plot
    plt.figure(figsize=(12, 8))
    cpu_data = None
    for result in all_results_total:
        if result is not None and result["exec_type"] == "class CPUExecutor":
            cpu_data = result["data"]
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
                label = executor_label(exec_type, lowp_type, use_wmma)
                times = [d["diff_ms"].iloc[-1] for d in df]
                speedup = [cpu / time if time > 0 else 0 for cpu, time in zip(cpu_times, times)]
                plt.plot(hidden_layers, speedup, label=label, marker='o', color=get_label_color(label))
                max_speedups[label] = {"max_speedup": max(speedup), "layer_size": hidden_layers[speedup.index(max(speedup))]}
                min_speedups[label] = {"min_speedup": min(speedup), "layer_size": hidden_layers[speedup.index(min(speedup))]}

    plt.xlabel("Hidden layer size")
    plt.ylabel("Speedup factor")
    plt.legend()
    plt.title("Speedup relative to CPU float baseline")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scaling_results_speedup.png")

    # Print speedup tables
    print("Max Speedup Information:")
    print(f"{'Configuration':<50} {'Max Speedup':<15} {'Layer Size':<15}")
    for label, info in max_speedups.items():
        print(f"{label:<50} {info['max_speedup']:<15.2f} {info['layer_size']:<15}")

    print("Minimum Speedup Information:")
    print(f"{'Configuration':<50} {'Min Speedup':<15} {'Layer Size':<15}")
    for label, info in min_speedups.items():
        print(f"{label:<50} {info['min_speedup']:<15.2f} {info['layer_size']:<15}")

    # WMMA vs CUDA Float per-layer speedup table
    # Find CUDA Float and WMMA Half datasets
    cuda_float = None
    wmma_half = None
    for result in all_results_total:
        if result is not None:
            if result["exec_type"] == "class CUDAExecutor" and result["lowp_type"] == "float" and result["use_wmma"] == 0:
                cuda_float = result
            if result["exec_type"] == "class CUDAExecutor" and result["lowp_type"] == "struct __half" and result["use_wmma"] == 1:
                wmma_half = result

    if cuda_float is not None and wmma_half is not None:
        # Intersect hidden layer sizes to ensure alignment
        hidden_cf = cuda_float["hidden_layers"]
        hidden_wm = wmma_half["hidden_layers"]
        common_hidden = [h for h in hidden_cf if h in hidden_wm]
        # Build maps from hidden size to final diff_ms value
        cf_map = {}
        for h, df in zip(cuda_float["hidden_layers"], cuda_float["data"]):
            if h in common_hidden:
                cf_map[h] = df["diff_ms"].iloc[-1]
        wm_map = {}
        for h, df in zip(wmma_half["hidden_layers"], wmma_half["data"]):
            if h in common_hidden:
                wm_map[h] = df["diff_ms"].iloc[-1]
        print("\nWMMA vs CUDA Float Speedup (CUDA Float time / WMMA Half time):")
        header = f"{'Hidden':<10} {'CUDA Float (ms)':<18} {'WMMA Half (ms)':<18} {'Speedup':<10}"
        print(header)
        print('-' * len(header))
        for h in common_hidden:
            cf_t = cf_map.get(h, None)
            wm_t = wm_map.get(h, None)
            if cf_t is None or wm_t is None or wm_t <= 0:
                continue
            speedup = cf_t / wm_t
            print(f"{h:<10} {cf_t:<18.2f} {wm_t:<18.2f} {speedup:<10.3f}")
    else:
        print("\nWMMA vs CUDA Float speedup table skipped (required data missing).")

    # Plot wmma vs cuda float speedup
    if cuda_float is not None and wmma_half is not None:
        plt.figure(figsize=(12, 8))
        speedups = []
        hidden_sizes = []
        for h in common_hidden:
            cf_t = cf_map.get(h, None)
            wm_t = wm_map.get(h, None)
            if cf_t is None or wm_t is None or wm_t <= 0:
                continue
            speedup = cf_t / wm_t
            hidden_sizes.append(h)
            speedups.append(speedup)
        plt.plot(hidden_sizes, speedups, marker='o', color='purple')
        plt.xlabel("Hidden layer size")
        plt.ylabel("Speedup (CUDA Float time / WMMA Half time)")
        plt.title("WMMA Half vs CUDA Float Speedup")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("wmma_vs_cuda_float_speedup.png")

    plt.show()

def plot_power_data(all_power_results):
    # Collect unique hidden layer sizes across all configurations
    all_hidden = set()
    for result in all_power_results:
        if result is not None:
            all_hidden.update(result["hidden_layers"])
    hidden_layers_sorted = sorted(all_hidden)
    x_indices = list(range(len(hidden_layers_sorted)))

    valid_results = [r for r in all_power_results if r is not None]
    n_cfg = len(valid_results)
    if n_cfg == 0:
        print("No power data to plot.")
        return

    bar_width = 0.8 / n_cfg

    # Build per-config energy maps
    config_energy_maps = []  # (label, cpu_map, gpu_map)
    plt.figure(figsize=(12, 8))
    for i, result in enumerate(valid_results):
        power_data = result["power_data"]
        use_wmma = result["use_wmma"]
        lowp_type = result["lowp_type"]
        exec_type = result["exec_type"]
        label = executor_label(exec_type, lowp_type, use_wmma)
        cpu_map = {d["hidden_layer_1"]: d["power_data"]["cpu_energy_Wh"] for d in power_data}
        gpu_map = {d["hidden_layer_1"]: d["power_data"]["gpu_energy_Wh"] for d in power_data}
        config_energy_maps.append((label, cpu_map, gpu_map))

        cpu_energies = [cpu_map.get(h, 0) for h in hidden_layers_sorted]
        gpu_energies = [gpu_map.get(h, 0) for h in hidden_layers_sorted]
        offsets = [x - 0.4 + bar_width/2 + i * bar_width for x in x_indices]
        plt.bar(offsets, cpu_energies, width=bar_width, label=f"{label} (CPU)", color=get_label_color(label), alpha=0.7)
        plt.bar(offsets, gpu_energies, width=bar_width, bottom=cpu_energies, label=f"{label} (GPU)", color=get_label_color(label), alpha=0.35, hatch='//')

    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Energy (Wh)")
    plt.title("Power Consumption (Stacked CPU + GPU Energy)")
    plt.xticks(x_indices, hidden_layers_sorted, rotation=45)
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    seen = set()
    filtered = []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            filtered.append((h, l))
            seen.add(l)
    plt.legend([h for h, _ in filtered], [l for _, l in filtered], fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("power_scaling_results_stacked.png")

    # Relative energy plot (stacked CPU/GPU percentages vs CPU baseline total energy)
    cpu_baseline_total = None
    cpu_baseline_label = None
    for label, cpu_map, gpu_map in config_energy_maps:
        if "CPU" in label and cpu_baseline_total is None:
            cpu_baseline_total = {h: cpu_map.get(h, 0) + gpu_map.get(h, 0) for h in hidden_layers_sorted}
            cpu_baseline_label = label

    if cpu_baseline_total is None:
        print("CPU baseline not found; skipping percentage plot.")
        plt.show()
        return

    plt.figure(figsize=(12, 8))
    for i, (label, cpu_map, gpu_map) in enumerate(config_energy_maps):
        cpu_perc = []
        gpu_perc = []
        total_perc = []
        for h in hidden_layers_sorted:
            base = cpu_baseline_total.get(h, 0)
            cpu_val = cpu_map.get(h, 0)
            gpu_val = gpu_map.get(h, 0)
            if base > 0:
                cpu_pct = cpu_val / base * 100.0
                gpu_pct = gpu_val / base * 100.0
            else:
                cpu_pct = 0.0
                gpu_pct = 0.0
            cpu_perc.append(cpu_pct)
            gpu_perc.append(gpu_pct)
            total_perc.append(cpu_pct + gpu_pct)
        offsets = [x - 0.4 + bar_width/2 + i * bar_width for x in x_indices]
        plt.bar(offsets, cpu_perc, width=bar_width, label=f"{label} (CPU %)", color=get_label_color(label), alpha=0.7)
        plt.bar(offsets, gpu_perc, width=bar_width, bottom=cpu_perc, label=f"{label} (GPU %)", color=get_label_color(label), alpha=0.35, hatch='//')
        # annotate total percent at top of stack
        for x_off, tot in zip(offsets, total_perc):
            plt.text(x_off, tot, f"{tot:.1f}%", ha='center', va='bottom', fontsize=8)

    plt.xlabel("Hidden layer size")
    plt.ylabel(f"Energy components (% of {cpu_baseline_label} total)")
    plt.title("CPU/GPU energy components relative to CPU float baseline")
    plt.xticks(x_indices, hidden_layers_sorted, rotation=45)
    plt.axhline(100, color='gray', linestyle='--', linewidth=1)
    # Consolidated legend without duplicates
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    seen = set()
    filtered = []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            filtered.append((h, l))
            seen.add(l)
    plt.legend([h for h, _ in filtered], [l for _, l in filtered], fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("power_scaling_results_percent_cpu.png")

    # Print table
    print("Power Consumption Data (Wh):")
    header = f"{'Config':<30} {'Layer':<8} {'CPU':>10} {'GPU':>10} {'Total':>10}"
    print(header)
    print('-' * len(header))
    for label, cpu_map, gpu_map in config_energy_maps:
        for h in hidden_layers_sorted:
            cpu_e = cpu_map.get(h, 0.0)
            gpu_e = gpu_map.get(h, 0.0)
            total_e = cpu_e + gpu_e
            if cpu_e == 0 and gpu_e == 0:
                continue
            print(f"{label:<30} {h:<8} {cpu_e:>10.4f} {gpu_e:>10.4f} {total_e:>10.4f}")

    plt.show()

if __name__ == "__main__":
    # Accuracy/time results
    directory = "logs/accuracy"
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
        "CPU, Float",
        "CUDA, Float",
        "CUDA, Half",
        "CUDA, Half, WMMA"
    ]
    plot_results(results_list, labels)

    # Threading results
    threading_results = load_threading_data()
    if threading_results:
        plot_threading_results(threading_results)

    # Scaling results
    scaling_directory = "logs/scaling"
    hidden_layer_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 28, 32, 64, 128, 256, 512, 768, 1024, 2048]
    scaling_data_cuda_float = load_scaling_data(scaling_directory, use_wmma=0, lowp_type="float", exec_type="class CUDAExecutor", hidden_layers=hidden_layer_sizes)
    scaling_data_cuda_wmma_half = load_scaling_data(scaling_directory, use_wmma=1, lowp_type="struct __half", exec_type="class CUDAExecutor", hidden_layers=hidden_layer_sizes)
    scaling_data_cpu_float = load_scaling_data(scaling_directory, use_wmma=0, lowp_type="float", exec_type="class CPUExecutor", hidden_layers=hidden_layer_sizes)
    all_results_total = [scaling_data_cuda_float, scaling_data_cuda_wmma_half, scaling_data_cpu_float]
    plot_scaling_data(all_results_total)

    # Power results
    power_directory = "C:\\Users\\mateu\\source\\repos\\DeepPotatoGPU\\demo\\digits\\charts\\logs\\power"
    power_hidden_layers = [16, 128, 512, 1024, 4096]
    power_data_cuda_float = load_all_power_data(power_directory, use_wmma=0, lowp_type="float", exec_type="class CUDAExecutor", hidden_layers=power_hidden_layers)
    power_data_cuda_half = load_all_power_data(power_directory, use_wmma=0, lowp_type="struct __half", exec_type="class CUDAExecutor", hidden_layers=power_hidden_layers)
    power_data_cuda_wmma_half = load_all_power_data(power_directory, use_wmma=1, lowp_type="struct __half", exec_type="class CUDAExecutor", hidden_layers=power_hidden_layers)
    power_data_cpu_float = load_all_power_data(power_directory, use_wmma=0, lowp_type="float", exec_type="class CPUExecutor", hidden_layers=power_hidden_layers)
    all_power_results = [power_data_cpu_float, power_data_cuda_float, power_data_cuda_half, power_data_cuda_wmma_half]
    plot_power_data(all_power_results)
