from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import datetime
import json
from tools import convert_pcap_to_features, FEATURE_LIST, METADATA_FEATURES
from pandas import DataFrame
import os
import pprint
import sys


def load_target_servers(json_file):
    """Load target server information from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def get_target_ips_for_file(filename, server_data):
    """Extract target server IPs for a specific PCAP filename from server data."""
    matching_ips = set()
    for entry in server_data:
        if isinstance(entry, dict) and entry.get('filename') == filename:
            server_ip = entry.get('server_ip')
            if server_ip:
                matching_ips.add(server_ip)
    return list(matching_ips) if matching_ips else []


def metrics_from_counts(tp, fp, tn, fn):
    total_records = tp + fp + tn + fn
    total_positives = tp + fn
    total_negatives = tn + fp

    tpr = tp / total_positives if total_positives else 0
    fpr = fp / total_negatives if total_negatives else 0

    return {
        "True_Positive_Rate": tpr,
        "False_Positive_Rate": fpr,
        "Precision": tp / (tp + fp) if (tp + fp) else 0,
        "F1_Score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0,
        "Total_Positives": total_positives,
        "Total_Negatives": total_negatives,
        "True_Positives": tp,
        "False_Positives": fp,
        "True_Negatives": tn,
        "False_Negatives": fn,
        "Specificity": tn / total_negatives if total_negatives else 0,
        "Negative_Predictive_Value": tn / (tn + fn) if (tn + fn) else 0,
        "Prevalence": (tp + fn) / total_records if total_records else 0,
        "Total_Records": total_records
    }


def score_dataframe(input_df, target_list):
    target_set = set(target_list)

    predicted_positive = input_df['predicted'].to_numpy(dtype=float) >= 0.5
    actual_positive = input_df['server_ip'].isin(target_set).to_numpy()

    return metrics_from_counts(
        tp=int(np.sum(predicted_positive & actual_positive)),
        fp=int(np.sum(predicted_positive & ~actual_positive)),
        tn=int(np.sum(~predicted_positive & ~actual_positive)),
        fn=int(np.sum(~predicted_positive & actual_positive))
    )


class NoUsableSessions(ValueError):
    """Feature extraction produced no sessions to evaluate."""


def process_file(input_file, model):
    input_data = convert_pcap_to_features(input_file)
    if len(input_data.index) == 0:
        raise NoUsableSessions("no usable TCP sessions were extracted")

    missing_columns = [
        column for column in [*METADATA_FEATURES, *FEATURE_LIST]
        if column not in input_data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{input_file}: extracted session data is missing required columns: "
            + ", ".join(missing_columns)
        )

    stream_metadata = input_data[METADATA_FEATURES].reset_index(drop=True)
    feature_data = input_data[FEATURE_LIST].reset_index(drop=True)

    # Sanity check: feature count must match model input
    assert feature_data.shape[1] == model.input_shape[-1], \
        "Feature count does not match model input shape"

    predictions = model.predict(feature_data, verbose=0)

    # Robust handling of NN output shape
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        # Sigmoid output
        predicted_probs = predictions[:, 0]
    elif predictions.ndim == 2 and predictions.shape[1] == 2:
        # Softmax output: positive class
        predicted_probs = predictions[:, 1]
    else:
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")

    stream_metadata['predicted'] = predicted_probs
    return DataFrame(stream_metadata)


def main():
    start = datetime.datetime.now()

    parser = ArgumentParser(
        description="Extract features from PCAP and score using trained neural network"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input PCAP file or directory")
    parser.add_argument("-m", "--model", required=True,
                        help="Saved TensorFlow model")
    parser.add_argument("-s", "--save",
                        help="Optional CSV file to save output results")
    parser.add_argument("-j", "--json", required=True,
                        help="JSON file containing target server information")

    args = parser.parse_args()

    # Load target servers from JSON file
    server_data = load_target_servers(args.json)
    print(f"Loaded {len(server_data)} server entries from {args.json}")

    model = tf.keras.models.load_model(args.model)

    if os.path.isfile(args.input):
        input_basename = os.path.basename(args.input)
        target_ips = get_target_ips_for_file(input_basename, server_data)
        if not target_ips:
            print(f"Error: No matching target IPs found for {input_basename} in {args.json}")
            exit(1)
        try:
            raw_results = process_file(args.input, model)
        except NoUsableSessions as exc:
            print(f"Warning: Skipping {input_basename}: {exc}. No results saved.", file=sys.stderr)
            raise SystemExit(1)
        results = score_dataframe(raw_results, target_ips)

        print(f"\nFilename: {args.input}")
        print(f"Target IPs for this file: {target_ips}")
        pprint.pprint(results, sort_dicts=False)

        if args.save:
            DataFrame([results]).to_csv(args.save, index=False)

    elif os.path.isdir(args.input):
        all_file_results = []
        processed_files = 0
        skipped_files = []

        for filename in sorted(os.listdir(args.input)):
            if filename.endswith(('.pcap', '.pcapng')):
                infile = os.path.join(args.input, filename)
                if not os.path.isfile(infile):
                    continue

                target_ips = get_target_ips_for_file(filename, server_data)
                if not target_ips:
                    print(
                        f"Warning: Skipping {filename}: no matching target IPs found in {args.json}",
                        file=sys.stderr,
                    )
                    skipped_files.append((filename, "no matching target IPs"))
                    continue

                try:
                    raw_results = process_file(infile, model)
                except NoUsableSessions as exc:
                    print(f"Warning: Skipping {filename}: {exc}", file=sys.stderr)
                    skipped_files.append((filename, str(exc)))
                    continue
                processed_files += 1

                file_results = score_dataframe(raw_results, target_ips)
                file_results['Filename'] = filename
                all_file_results.append(file_results)
                print(f"\nFilename: {filename}")
                print(f"Target IPs: {target_ips}")
                pprint.pprint(file_results, sort_dicts=False)

        if skipped_files:
            print(f"\nSkipped {len(skipped_files)} file(s) (excluded from metrics):")
            for filename, reason in skipped_files:
                print(f"  {filename}: {reason}")

        if not all_file_results:
            if skipped_files:
                print("No files were evaluated; all captures were skipped. No results saved.")
            else:
                print("No .pcap or .pcapng files found in input directory.")
        else:
            aggregate_results = metrics_from_counts(
                tp=sum(f['True_Positives'] for f in all_file_results),
                fp=sum(f['False_Positives'] for f in all_file_results),
                tn=sum(f['True_Negatives'] for f in all_file_results),
                fn=sum(f['False_Negatives'] for f in all_file_results)
            )
            aggregate_results['Filename'] = 'AGGREGATE'

            print(f"\nAggregate statistics across {processed_files} files:")
            pprint.pprint(aggregate_results, sort_dicts=False)

            if args.save:
                DataFrame(all_file_results + [aggregate_results]).to_csv(args.save, index=False)

    elapsed = datetime.datetime.now() - start
    print(f"Elapsed time: {elapsed}")


if __name__ == "__main__":
    main()
