"""
Live TLS-session classifier.

Runs a 30-second capture window on a network interface, extracts session
features, scores them with a Keras model, and writes two outputs:

  * ledger file       — one line per classification cycle, holding the
                        Unix timestamp at which that cycle began.
  * classification file — one line per (client_ip, server_ip) pair that
                        produced at least one positive session in the
                        cycle, with the cycle's unix timestamp and the
                        minimum probability seen for that pair.

The loop runs indefinitely; transient errors are logged to stderr and the
loop continues on the next cycle.
"""

from argparse import ArgumentParser
from datetime import datetime, timedelta
import csv
import os
import signal
import sys
import time
import traceback

import tensorflow as tf
from tcp_sessions_ext import TCPSessionExtractor

from live_tools import FEATURE_LIST, convert_sessions_to_features

CYCLE_SECONDS = 30
LEDGER_HEADER = ["unix_timestamp", "sessions_seen", "positives"]
CLASSIFICATION_HEADER = [
    "unix_timestamp",
    "client_ip",
    "server_ip",
    "positive_count",
    "min_probability",
]

_RUNNING = True


def _handle_sigterm(_signum, _frame):
    global _RUNNING
    _RUNNING = False


def _open_csv(path, header):
    """Open path in append mode, writing the header if the file is new/empty."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(header)
        fh.flush()
    return fh, writer


def _rotate_csv(fh, path, header):
    """Close `fh`, rename `path` with a timestamp suffix, reopen a fresh file."""
    fh.close()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        root, ext = os.path.splitext(path)
        rotated = f"{root}-{stamp}{ext}"
        # Avoid clobbering if the target already exists.
        i = 1
        while os.path.exists(rotated):
            rotated = f"{root}-{stamp}.{i}{ext}"
            i += 1
        os.rename(path, rotated)
    return _open_csv(path, header)


def _next_midnight_epoch(now=None):
    """Return epoch seconds for the next local midnight after `now`."""
    if now is None:
        now = datetime.now()
    tomorrow = (now + timedelta(days=1)).date()
    next_midnight = datetime.combine(tomorrow, datetime.min.time())
    return int(next_midnight.timestamp())


def run_cycle(interface, bpf_filter, model, threshold, cycle_start_ts):
    """Capture for CYCLE_SECONDS on `interface`, classify, return (sessions_seen,
    list[positive-row-dict])."""
    extractor = TCPSessionExtractor(
        interface=interface,
        bpf_filter=bpf_filter,
        live_timeout=float(CYCLE_SECONDS),
        max_session_duration_sec=None,
        max_session_packets=10,
    )

    df, stats = convert_sessions_to_features(extractor)
    if df.empty:
        return stats["sessions_seen"], []

    features = df[FEATURE_LIST].reset_index(drop=True)
    predictions = model.predict(features, verbose=0)
    if predictions.ndim != 2 or predictions.shape[1] != 1:
        raise ValueError(f"Unexpected model output shape {predictions.shape}")

    probs = predictions[:, 0]
    df = df.assign(probability=probs, label=(probs >= threshold).astype(int))
    positives = df[df["label"] == 1]
    if positives.empty:
        return stats["sessions_seen"], []

    grouped = positives.groupby(["client_ip", "server_ip"], as_index=False).agg(
        positive_count=("label", "sum"),
        min_probability=("probability", "min"),
    )

    rows = []
    for _, r in grouped.iterrows():
        rows.append(
            {
                "unix_timestamp": cycle_start_ts,
                "client_ip": r["client_ip"],
                "server_ip": r["server_ip"],
                "positive_count": int(r["positive_count"]),
                "min_probability": float(r["min_probability"]),
            }
        )
    return stats["sessions_seen"], rows


def main():
    parser = ArgumentParser(description="Live TLS-session classifier")
    parser.add_argument("-i", "--interface", required=True,
                        help="Network interface to capture on (e.g. en0)")
    parser.add_argument("-m", "--model", required=True,
                        help="Path to the Keras model")
    parser.add_argument("-l", "--ledger", required=True,
                        help="Path to the ledger CSV (append-only)")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the classification CSV (append-only)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-f", "--bpf-filter", default="tcp and port 443",
                        help="BPF filter for live capture (default: 'tcp and port 443')")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigterm)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    model = tf.keras.models.load_model(args.model)

    ledger_fh, ledger_writer = _open_csv(args.ledger, LEDGER_HEADER)
    class_fh, class_writer = _open_csv(args.output, CLASSIFICATION_HEADER)
    next_rotation = _next_midnight_epoch()

    try:
        while _RUNNING:
            cycle_start = int(time.time())

            if cycle_start >= next_rotation:
                ledger_fh, ledger_writer = _rotate_csv(ledger_fh, args.ledger, LEDGER_HEADER)
                class_fh, class_writer = _rotate_csv(class_fh, args.output, CLASSIFICATION_HEADER)
                next_rotation = _next_midnight_epoch()

            sessions_seen = 0
            positive_rows = []
            try:
                sessions_seen, positive_rows = run_cycle(
                    args.interface, args.bpf_filter, model, args.threshold, cycle_start,
                )
            except Exception:
                print(f"[{cycle_start}] cycle failed:", file=sys.stderr)
                traceback.print_exc()

            ledger_writer.writerow([cycle_start, sessions_seen, len(positive_rows)])
            ledger_fh.flush()

            for row in positive_rows:
                class_writer.writerow([row[c] for c in CLASSIFICATION_HEADER])
            if positive_rows:
                class_fh.flush()

            # If the cycle finished early (e.g. immediate error), pace the loop
            # so we still run at most one cycle per CYCLE_SECONDS window.
            elapsed = time.time() - cycle_start
            if elapsed < CYCLE_SECONDS and _RUNNING:
                time.sleep(CYCLE_SECONDS - elapsed)
    finally:
        ledger_fh.close()
        class_fh.close()


if __name__ == "__main__":
    main()
