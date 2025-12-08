import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams["font.size"] = 15
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

def executor_label(exec_type, data_type, use_wmma):
    label = exec_type.replace("class ", "").replace("Executor", "").strip()
    if data_type == "float":
        label += ", Float"
    else:
        label += ", Half"
    if use_wmma:
        label += ", Tensor Cores"
    return label

def get_label_color(label):
    if "CPU" in label:
        return "blue"
    elif "CUDA" in label and "Float" in label:
        return "orange"
    elif "CUDA" in label and "Half" in label and "Tensor Cores" not in label:
        return "green"
    elif "CUDA" in label and "Half" in label and "Tensor Cores" in label:
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
    # check if the lowp_type is 'struct __half' or 'float'
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

def load_time_data(directory, use_wmma, lowp_type, exec_type, threads = -1, hidden_layer_1 = -1):
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
        print("No results. end_of_filename:", end_of_filename)
        return None
    combined_df = pd.concat(all_results)
    grouped = combined_df.groupby("samplesTotal")
    averaged_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index()
    for col in ["diff_ms", "forwardAvg", "backwardAvg", "updateAvg"]:
        averaged_df[col + "_std"] = std_df[col]
    averaged_df = averaged_df.drop(columns = ["testAccuracy", "loss"])
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
        print("No accuracy results found. end_of_filename:", end_of_filename)
        return None
    combined_df = pd.concat(all_results)
    grouped_df = combined_df.groupby("samplesTotal")
    averaged_df = grouped_df.mean().reset_index()
    std_df = grouped_df.std().reset_index()
    for col in ["testAccuracy", "loss"]:
        averaged_df[col + "_std"] = std_df[col]
    return averaged_df

def load_and_average_results(directory, use_wmma, lowp_type, exec_type, threads = -1):
    averaged_time_df = load_time_data(directory, use_wmma, lowp_type, exec_type, threads)
    averaged_accuracy_df = load_accuracy_loss_data(directory, use_wmma, lowp_type, exec_type, threads)

    if averaged_time_df is None or averaged_accuracy_df is None:
        return None

    merged_df = averaged_time_df.merge(averaged_accuracy_df[["samplesTotal", "testAccuracy", "loss","testAccuracy_std", "loss_std"]], on = "samplesTotal", how = "left")

    return merged_df

def load_threading_data(directory):
    print("Loading threading data...")
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
            df = pd.read_csv(filepath, sep = ",")
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
        print("No power results found. end_of_filename:", end_of_filename)
        return None
    combined_df = pd.DataFrame(all_results)
    averaged_df = combined_df.mean().to_dict()
    std_df = combined_df.std().to_dict()
    return {"mean": averaged_df, "std": std_df}

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

    # time metrics of full runs
    plt.figure(figsize = (12, 8))
    bar_width = 0.15
    groups = ["diff_ms", "forwardAvg", "backwardAvg", "updateAvg"]
    group_labels = ["Całkowity czas", "Propagacja w przód", "Propagacja wstecz", "Aktualizacja"]
    x = range(len(groups))
    for i, (df, label) in enumerate(zip(results_list, labels)):
        if df is not None:
            offset = i * bar_width
            means = df[groups].iloc[-1].values / 1000.0
            stds = df[[g + "_std" for g in groups]].iloc[-1].values / 1000.0
            plt.bar(
                [x_val + offset for x_val in x],
                means,
                yerr = stds,
                width = bar_width,
                label = label,
                color = get_label_color(label),
                error_kw = {"elinewidth": 1, "capsize": 3, "capthick": 1}
            )
    plt.title("Czas treningu")
    plt.ylabel("Czas (s)")
    plt.xticks([r + bar_width * (len(results_list) - 1) / 2 for r in x], group_labels)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_metrics_full_runs.png")

    # print time metrics table
    print("Time metrics of full runs")
    print(f"{'Configuration':<30} {'Total Time':<20} {'Forward Pass':<20} {'Backward Pass':<20} {'Update':<20}")
    for df, label in zip(results_list, labels):
        if df is not None:
            total_mean = df["diff_ms"].iloc[-1] / 1000.0
            forward_mean = df["forwardAvg"].iloc[-1] / 1000.0
            backward_mean = df["backwardAvg"].iloc[-1] / 1000.0
            update_mean = df["updateAvg"].iloc[-1] / 1000.0

            total_std = df["diff_ms_std"].iloc[-1] / 1000.0
            forward_std = df["forwardAvg_std"].iloc[-1] / 1000.0
            backward_std = df["backwardAvg_std"].iloc[-1] / 1000.0
            update_std = df["updateAvg_std"].iloc[-1] / 1000.0

            print(
                f"{label:<30} "
                f"{total_mean:>6.2f} +- {total_std:<10.2f} "
                f"{forward_mean:>6.2f} +- {forward_std:<10.2f} "
                f"{backward_mean:>6.2f} +- {backward_std:<10.2f} "
                f"{update_mean:>6.2f} +- {update_std:<10.2f}"
            )

    # print time metrics as speedup multipliers
    print("\nTime metrics as speedup multipliers of CPU float:")
    cpu_float_df = results_list[0]  # Assuming first is CPU Float
    for df, label in zip(results_list, labels):
        if df is not None and cpu_float_df is not None:
            total_time_ratio = (cpu_float_df["diff_ms"].iloc[-1] / df["diff_ms"].iloc[-1])
            forward_time_ratio = (cpu_float_df["forwardAvg"].iloc[-1] / df["forwardAvg"].iloc[-1])
            backward_time_ratio = (cpu_float_df["backwardAvg"].iloc[-1] / df["backwardAvg"].iloc[-1])
            update_time_ratio = (cpu_float_df["updateAvg"].iloc[-1] / df["updateAvg"].iloc[-1])
            print(f"{label:<30} {total_time_ratio:<15.2f} {forward_time_ratio:<15.2f} {backward_time_ratio:<15.2f} {update_time_ratio:<15.2f}")

    # test accuracy over samples total
    plt.figure(figsize = (12, 8))
    for df, label in zip(results_list, labels):
        if df is not None:
            plt.plot(df["samplesTotal"], df["testAccuracy"] * 100, label = label, color = get_label_color(label))
    plt.title("Dokładność predykcji w zależności od liczby próbek")
    plt.xlabel("Liczba próbek")
    plt.ylabel("Dokładność predykcji (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_accuracy_over_samples.png")

    # table of accuracy for selected sample counts
    sample_counts = [100000, 300000, 600000]
    print("\nTest accuracy at selected sample counts:")
    print(f"{'Configuration':<30} " + " ".join([f"{sc:<25}" for sc in sample_counts]))
    for df, label in zip(results_list, labels):
        if df is not None:
            accuracies = []
            for sc in sample_counts:
                closest_idx = (df["samplesTotal"] - sc).abs().idxmin()
                accuracy = df["testAccuracy"].iloc[closest_idx]
                accuracy_std = df["testAccuracy_std"].iloc[closest_idx] if "testAccuracy_std" in df.columns else 0.0
                accuracies.append(f"{accuracy * 100:.2f}% +- {accuracy_std * 100:.2f}%")
            print(f"{label:<30} " + " ".join([f"{acc:<25}" for acc in accuracies]))

    # test accuracy over time
    plt.figure(figsize = (12, 8))
    for df, label in zip(results_list, labels):
        if df is not None:
            plt.plot(df["diff_ms"], df["testAccuracy"] * 100, label = label, color = get_label_color(label))
    plt.title("Dokładność predykcji w zależności od czasu treningu")
    plt.xlabel("Czas (ms)")
    plt.ylabel("Dokładność predykcji (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_accuracy_over_time.png")

    plt.show()

def plot_threading_results(threading_results):
    plt.figure(figsize = (12, 8))
    x_values = [tr[0] for tr in threading_results]
    y_values = [tr[1]["diff_ms"].iloc[-1] / 1000 for tr in threading_results]
    y_err = [
        (tr[1]["diff_ms_std"].iloc[-1] / 1000) if "diff_ms_std" in tr[1].columns else 0 for tr in threading_results
    ]
    plt.errorbar(x_values, y_values, yerr = y_err, label="Time (s)", marker = 'o', linestyle = '-', capsize = 3)
    plt.xlabel("Number of threads")
    plt.ylabel("Time (s)")
    plt.title("Training time with multithreading")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("threading_results.png")

    # print threading results table with speedup
    plt.figure(figsize = (12, 8))
    speedups = []
    speedup_err = []
    print("Threading Results:")
    print(f"{'Threads':<10} {'Time (s)':<15} {'Speedup':<15}")
    for i, tr in enumerate(threading_results):
        if tr is not None:
            time = tr[1]["diff_ms"].iloc[-1] / 1000
            time_std = tr[1]["diff_ms_std"].iloc[-1] / 1000
            speedup = (threading_results[0][1]["diff_ms"].iloc[-1] / 1000) / time
            T_ref = threading_results[0][1]["diff_ms"].iloc[-1] / 1000
            s_err = (T_ref / time) * (time_std / time) if time > 0 else 0.0
            
            print(f"{i:<10} {time:<15.2f} +- {time_std:<10.2f} {speedup:<15.2f} +- {s_err:<10.2f}")
            speedups.append(speedup)
            speedup_err.append(s_err)

    plt.errorbar(x_values, speedups, yerr = speedup_err, label = "Współczynnik przyspieszenia", marker = 'o', linestyle = '-', color = 'orange', capsize = 3)
    plt.xlabel("Liczba wątków")
    plt.ylabel("Współczynnik przyspieszenia")
    plt.title("Współczynnik przyspieszenia treningu w zależności od liczby wątków CPU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("threading_speedup.png")
    plt.show()

def plot_scaling_data(all_results_total):
    plt.figure(figsize = (12, 8))
    for result in all_results_total:
        if result is not None:
            hidden_layers = result["hidden_layers"]
            use_wmma = result["use_wmma"]
            lowp_type = result["lowp_type"]
            exec_type = result["exec_type"]
            df = result["data"]
            label = executor_label(exec_type, lowp_type, use_wmma)
            y_err = [
                (d["diff_ms_std"].iloc[-1] / 1000) if "diff_ms_std" in d.columns else 0
                for d in df
            ]
            plt.errorbar(hidden_layers, [d["diff_ms"].iloc[-1] / 1000 for d in df], yerr = y_err, label = label, marker = 'o', color = get_label_color(label))
    plt.xlabel("Rozmiar warstwy ukrytej")
    plt.ylabel("Czas (s)")
    plt.title("Czas treningu w zależności od rozmiaru warstwy ukrytej")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scaling_results.png")

    # speedup factor plot
    plt.figure(figsize = (12, 8))
    cpu_data = None
    for result in all_results_total:
        if result is not None and result["exec_type"] == "class CPUExecutor":
            cpu_data = result["data"]
            break

    max_speedups = {}
    min_speedups = {}
    if cpu_data is not None:
        cpu_times = [d["diff_ms"].iloc[-1] for d in cpu_data]
        cpu_stds = [d["diff_ms_std"].iloc[-1] if "diff_ms_std" in d.columns else 0.0 for d in cpu_data]
        for result in all_results_total:
            if result is not None:
                hidden_layers = result["hidden_layers"]
                use_wmma = result["use_wmma"]
                lowp_type = result["lowp_type"]
                exec_type = result["exec_type"]
                df = result["data"]
                label = executor_label(exec_type, lowp_type, use_wmma)
                times = [d["diff_ms"].iloc[-1] for d in df]
                time_stds = [d["diff_ms_std"].iloc[-1] if "diff_ms_std" in d.columns else 0.0 for d in df]
                speedup = []
                speedup_err = []
                for cpu_t, cpu_s, t, s in zip(cpu_times, cpu_stds, times, time_stds):
                    if t > 0 and cpu_t > 0:
                        val = cpu_t / t
                        # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
                        # assume two measurements are uncorrelated
                        err = val * ((cpu_s / cpu_t)**2 + (s / t)**2) ** 0.5
                    else:
                        val = 0.0
                        err = 0.0
                    speedup.append(val)
                    speedup_err.append(err)
                plt.errorbar(hidden_layers, speedup, yerr = speedup_err, label = label, marker = 'o', linestyle = '-', color = get_label_color(label), capsize = 3)
                max_speedups[label] = {"max_speedup": max(speedup), "layer_size": hidden_layers[speedup.index(max(speedup))], "max_time": times[speedup.index(max(speedup))], "max_time_std": time_stds[speedup.index(max(speedup))], "max_speedup_std": speedup_err[speedup.index(max(speedup))]}
                min_speedups[label] = {"min_speedup": min(speedup), "layer_size": hidden_layers[speedup.index(min(speedup))], "min_time": times[speedup.index(min(speedup))], "min_time_std": time_stds[speedup.index(min(speedup))], "min_speedup_std": speedup_err[speedup.index(min(speedup))]}

    plt.xlabel("Rozmiar warstwy ukrytej")
    plt.ylabel("Współczynnik przyspieszenia")
    plt.legend()
    plt.title("Przyspieszenie względem CPU float")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scaling_results_speedup.png")

    # Print speedup tables
    print("Max Speedup Information:")
    print(f"{'Configuration':<50} {'Max Speedup':<15} {'Layer Size':<15} {'Max Time (s)':<15} {'Max Time Std (s)':<15} {'Max Speedup Std':<15}")
    for label, info in max_speedups.items():
        print(f"{label:<50} {info['max_speedup']:<15.2f} {info['layer_size']:<15} {info['max_time'] / 1000:<15.2f} {info['max_time_std'] / 1000:<15.2f} {info['max_speedup_std']:<15.2f}")

    print("Minimum Speedup Information:")
    print(f"{'Configuration':<50} {'Min Speedup':<15} {'Layer Size':<15} {'Min Time (s)':<15} {'Min Time Std (s)':<15} {'Min Speedup Std':<15}")
    for label, info in min_speedups.items():
        print(f"{label:<50} {info['min_speedup']:<15.2f} {info['layer_size']:<15} {info['min_time'] / 1000:<15.2f} {info['min_time_std'] / 1000:<15.2f} {info['min_speedup_std']:<15.2f}")

    # WMMA vs CUDA Float per-layer speedup table
    cuda_float = None
    wmma_half = None
    for result in all_results_total:
        if result is not None:
            if result["exec_type"] == "class CUDAExecutor" and result["lowp_type"] == "float" and result["use_wmma"] == 0:
                cuda_float = result
            if result["exec_type"] == "class CUDAExecutor" and result["lowp_type"] == "struct __half" and result["use_wmma"] == 1:
                wmma_half = result

    if cuda_float is not None and wmma_half is not None:
        hidden_cf = cuda_float["hidden_layers"]
        hidden_wm = wmma_half["hidden_layers"]
        common_hidden = [h for h in hidden_cf if h in hidden_wm]
        cf_map = {}
        for h, df in zip(cuda_float["hidden_layers"], cuda_float["data"]):
            if h in common_hidden:
                cf_map[h] = df["diff_ms"].iloc[-1]
        wm_map = {}
        for h, df in zip(wmma_half["hidden_layers"], wmma_half["data"]):
            if h in common_hidden:
                wm_map[h] = df["diff_ms"].iloc[-1]
        print("\nWMMA vs CUDA Float speedup (CUDA Float time / WMMA Half time):")
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
        print("\nWMMA vs CUDA Float speedup table skipped")

    plt.show()

def plot_power_data(all_power_results):
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

    config_energy_maps = []
    plt.figure(figsize = (12, 8))
    for i, result in enumerate(valid_results):
        power_data = result["power_data"]
        use_wmma = result["use_wmma"]
        lowp_type = result["lowp_type"]
        exec_type = result["exec_type"]
        label = executor_label(exec_type, lowp_type, use_wmma)
        cpu_map_mean = {d["hidden_layer_1"]: d["power_data"]["mean"]["cpu_energy_Wh"] for d in power_data}
        gpu_map_mean = {d["hidden_layer_1"]: d["power_data"]["mean"]["gpu_energy_Wh"] for d in power_data}
        total_map_mean = {d["hidden_layer_1"]: d["power_data"]["mean"]["total_energy_Wh"] for d in power_data}
        cpu_map_std = {d["hidden_layer_1"]: d["power_data"]["std"]["cpu_energy_Wh"] for d in power_data}
        gpu_map_std = {d["hidden_layer_1"]: d["power_data"]["std"]["gpu_energy_Wh"] for d in power_data}
        total_map_std = {d["hidden_layer_1"]: d["power_data"]["std"]["total_energy_Wh"] for d in power_data}
        config_energy_maps.append((label, cpu_map_mean, gpu_map_mean, total_map_mean, cpu_map_std, gpu_map_std, total_map_std))

        cpu_energies = [cpu_map_mean.get(h, 0) for h in hidden_layers_sorted]
        gpu_energies = [gpu_map_mean.get(h, 0) for h in hidden_layers_sorted]
        cpu_stds = [cpu_map_std.get(h, 0) for h in hidden_layers_sorted]
        gpu_stds = [gpu_map_std.get(h, 0) for h in hidden_layers_sorted]
        offsets = [x - 0.4 + bar_width / 2 + i * bar_width for x in x_indices]
        plt.bar(offsets, cpu_energies, yerr = cpu_stds, width = bar_width, label = f"{label} (CPU)", color=get_label_color(label), alpha = 0.7, error_kw = {"elinewidth":1, "capsize":3, "capthick":1})
        if any(gpu_energies):
            plt.bar(offsets, gpu_energies, yerr=gpu_stds, width=bar_width, bottom=cpu_energies, label = f"{label} (GPU)", color=get_label_color(label), alpha = 0.35, hatch = '//', error_kw = {"elinewidth": 1, "capsize": 3, "capthick": 1})

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

    # relative energy plot
    cpu_baseline_total = None
    cpu_baseline_label = None
    for label, cpu_map, gpu_map, total_map, _, _, _ in config_energy_maps:
        if "CPU" in label and cpu_baseline_total is None:
            cpu_baseline_total = {h: cpu_map.get(h, 0) + gpu_map.get(h, 0) for h in hidden_layers_sorted}
            cpu_baseline_label = label

    if cpu_baseline_total is None:
        print("CPU baseline not found; skipping percentage plot.")
        plt.show()
        return

    plt.figure(figsize=(12, 8))
    for i, (label, cpu_map, gpu_map, total_map, cpu_std_map, gpu_std_map, total_std_map) in enumerate(config_energy_maps):
        cpu_perc = []
        gpu_perc = []
        total_perc = []
        cpu_perc_std = []
        gpu_perc_std = []
        total_perc_std = []
        for h in hidden_layers_sorted:
            base = cpu_baseline_total.get(h, 0)
            cpu_val = cpu_map.get(h, 0)
            gpu_val = gpu_map.get(h, 0)
            cpu_std = cpu_std_map.get(h, 0)
            gpu_std = gpu_std_map.get(h, 0)
            total_std = total_std_map.get(h, 0)
            if base > 0:
                cpu_pct = cpu_val / base * 100.0
                gpu_pct = gpu_val / base * 100.0
                cpu_pct_std = cpu_std / base * 100.0
                gpu_pct_std = gpu_std / base * 100.0
                total_pct_std = total_std / base * 100.0
            else:
                cpu_pct = 0.0
                gpu_pct = 0.0
                cpu_pct_std = 0.0
                gpu_pct_std = 0.0
                total_pct_std = 0.0
            cpu_perc.append(cpu_pct)
            gpu_perc.append(gpu_pct)
            total_perc.append(cpu_pct + gpu_pct)
            cpu_perc_std.append(cpu_pct_std)
            gpu_perc_std.append(gpu_pct_std)
            total_perc_std.append(total_pct_std)
        offsets = [x - 0.4 + bar_width / 2 + i * bar_width for x in x_indices]
        plt.bar(offsets, cpu_perc, width = bar_width, label = f"{label} (CPU %)", color = get_label_color(label), alpha = 0.7)
        plt.bar(offsets, gpu_perc, width = bar_width, bottom = cpu_perc, label = f"{label} (GPU %)", color = get_label_color(label), alpha = 0.35, hatch = '//')
        for x_off, tot in zip(offsets, total_perc):
            plt.text(x_off, tot, f"{tot:.1f}%", ha = 'center', va = 'bottom', fontsize = 8)
        plt.errorbar(offsets, total_perc, yerr = total_perc_std, fmt = 'none', ecolor = 'gray', elinewidth = 1, capsize = 3, capthick = 1)

    plt.xlabel("Rozmiar warstwy ukrytej")
    plt.ylabel(f"Procent zużytej energii względem {cpu_baseline_label}")
    plt.title(f"Procent zużytej energii względem {cpu_baseline_label}")
    plt.xticks(x_indices, hidden_layers_sorted, rotation = 45)
    plt.axhline(100, color = 'gray', linestyle = '--', linewidth = 1)

    handles, labels_leg = plt.gca().get_legend_handles_labels()
    seen = set()
    filtered = []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            filtered.append((h, l))
            seen.add(l)
    plt.legend([h for h, _ in filtered], [l for _, l in filtered], fontsize = 9)
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.6)
    plt.tight_layout()
    plt.savefig("power_scaling_results_percent_cpu.png")

    # print table
    print("Power Consumption Data (Wh) with std:")
    header = f"{'Config':<30} {'Layer':<8} {'CPU (Wh)':>18} {'GPU (Wh)':>18} {'Total (Wh)':>18}"
    print(header)
    print('-' * len(header))
    for label, cpu_map, gpu_map, total_map, cpu_std_map, gpu_std_map, total_std_map in config_energy_maps:
        for h in hidden_layers_sorted:
            cpu_e = cpu_map.get(h, 0.0)
            gpu_e = gpu_map.get(h, 0.0)
            total_e = total_map.get(h, 0.0)
            cpu_std = cpu_std_map.get(h, 0.0)
            gpu_std = gpu_std_map.get(h, 0.0)
            total_std = total_std_map.get(h, 0.0)
            if cpu_e == 0 and gpu_e == 0:
                continue
            print(
                f"{label:<30} {h:<8} "
                f"{cpu_e:>6.4f} +- {cpu_std:<10.4f} "
                f"{gpu_e:>6.4f} +- {gpu_std:<10.4f} "
                f"{total_e:>6.4f} +- {total_std:<10.4f}"
            )

    plt.show()

if __name__ == "__main__":
    # accuracy/time results
    directory = "logs\\accuracy"
    averaged_results_cpu_float = load_and_average_results(directory, use_wmma = 0, lowp_type = "float", exec_type = "class CPUExecutor")
    averaged_results_cuda_float = load_and_average_results(directory, use_wmma = 0, lowp_type = "float", exec_type = "class CUDAExecutor")
    averaged_results_cuda_half = load_and_average_results(directory, use_wmma = 0, lowp_type = "struct __half", exec_type = "class CUDAExecutor")
    averaged_results_wmma_half = load_and_average_results(directory, use_wmma = 1, lowp_type = "struct __half", exec_type = "class CUDAExecutor")

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
        "CUDA, Half, Tensor Cores"
    ]
    plot_results(results_list, labels)

    # threading results
    threading_results = load_threading_data(directory = "logs\\threading")
    if threading_results:
        plot_threading_results(threading_results)

    # scaling results
    scaling_directory = "logs\\scaling"
    hidden_layer_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 28, 32, 64, 128, 256, 512, 768, 1024, 2048]
    scaling_data_cuda_float = load_scaling_data(scaling_directory, use_wmma = 0, lowp_type = "float", exec_type = "class CUDAExecutor", hidden_layers = hidden_layer_sizes)
    scaling_data_cuda_wmma_half = load_scaling_data(scaling_directory, use_wmma = 1, lowp_type = "struct __half", exec_type = "class CUDAExecutor", hidden_layers = hidden_layer_sizes)
    scaling_data_cuda_half = load_scaling_data(scaling_directory, use_wmma = 0, lowp_type = "struct __half", exec_type = "class CUDAExecutor", hidden_layers = hidden_layer_sizes)
    scaling_data_cpu_float = load_scaling_data(scaling_directory, use_wmma = 0, lowp_type = "float", exec_type = "class CPUExecutor", hidden_layers = hidden_layer_sizes)
    all_results_total = [scaling_data_cuda_float, scaling_data_cuda_wmma_half, scaling_data_cuda_half, scaling_data_cpu_float]
    plot_scaling_data(all_results_total)

    # power results
    power_directory = "logs\\power"
    power_hidden_layers = [32, 128, 512, 1024, 2048]
    power_data_cuda_float = load_all_power_data(power_directory, use_wmma = 0, lowp_type = "float", exec_type = "class CUDAExecutor", hidden_layers = power_hidden_layers)
    power_data_cuda_half = load_all_power_data(power_directory, use_wmma = 0, lowp_type = "struct __half", exec_type = "class CUDAExecutor", hidden_layers = power_hidden_layers)
    power_data_cuda_wmma_half = load_all_power_data(power_directory, use_wmma = 1, lowp_type = "struct __half", exec_type = "class CUDAExecutor", hidden_layers = power_hidden_layers)
    power_data_cpu_float = load_all_power_data(power_directory, use_wmma = 0, lowp_type = "float", exec_type = "class CPUExecutor", hidden_layers = power_hidden_layers)
    all_power_results = [power_data_cpu_float, power_data_cuda_float, power_data_cuda_half, power_data_cuda_wmma_half]
    plot_power_data(all_power_results)
