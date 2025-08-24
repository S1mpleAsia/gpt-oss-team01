#!/usr/bin/env python3
from collections import defaultdict
import re
import sys


def analyze_log(log_lines):
    """
    Parses log lines to extract layer timing information and calculates statistics.

    Args:
        log_lines: An iterable of strings, where each string is a line from the log.

    Returns:
        A dictionary mapping layer names to their performance statistics.
    """
    # Regex to capture the layer name and the execution time in milliseconds.
    # It looks for "LayerName: 123.456 ms"
    log_pattern = re.compile(r"([\w\d_]+):\s+([\d\.]+\d*)\s+ms")

    # Use defaultdict to easily append timings to a list for each layer.
    layer_timings = defaultdict(list)

    for line in log_lines:
        match = log_pattern.match(line.strip())
        if match:
            layer_name, time_str = match.groups()
            try:
                # Store the time as a float
                time_ms = float(time_str)
                layer_timings[layer_name].append(time_ms)
            except ValueError:
                # Ignore lines where the time is not a valid float
                continue

    # --- Calculate Statistics ---
    stats = {}
    for name, timings in layer_timings.items():
        if not timings:
            continue

        count = len(timings)
        total_time = sum(timings)
        avg_time = total_time / count
        max_time = max(timings)
        min_time = min(timings)

        stats[name] = {
            "count": count,
            "min_time": min_time,
            "avg_time": avg_time,
            "max_time": max_time,
            "total_time": total_time,
        }

    return stats


def print_stats_table(stats):
    """
    Prints the calculated statistics in a formatted table.
    """
    if not stats:
        print("No timing data found in the input.")
        return

    # Sort layers by total time spent, descending
    sorted_layers = sorted(stats.items(),
                           key=lambda item: item[1]['total_time'],
                           reverse=True)

    # --- Print Header ---
    print("-" * 100)
    print(
        f"{'Layer Name':<25} | {'Count':>10} | {'Min Time (ms)':>15} | {'Max Time (ms)':>15} | {'Avg Time (ms)':>15} | {'Total Time (ms)':>15}"
    )
    print("-" * 100)

    # --- Print Rows ---
    for name, data in sorted_layers:
        print(
            f"{name:<25} | {data['count']:>10} | {data['min_time']:>15.5f} | {data['max_time']:>15.5f} | {data['avg_time']:>15.5f} | {data['total_time']:>15.5f}"
        )

    print("-" * 100)


if __name__ == "__main__":
    # Check if a log file path is provided as a command-line argument.
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_log_file>", file=sys.stderr)
        sys.exit(1)

    log_file_path = sys.argv[1]

    try:
        # Open and read the log file.
        with open(log_file_path, 'r') as f:
            log_data = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}",
              file=sys.stderr)
        sys.exit(1)

    # Analyze and print the results.
    layer_stats = analyze_log(log_data)
    print_stats_table(layer_stats)
