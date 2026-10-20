"""
Live-capture variant of tools.py.

Differs from tools.py in one respect: feature extraction operates on an
already-constructed iterator of sessions (typically a TCPSessionExtractor
opened against a live interface with a live_timeout) rather than opening a
pcap file. The FFT helper and feature list are re-used unchanged.
"""

from pandas import DataFrame
import numpy as np

from tools import (
    FEATURE_LIST,
    METADATA_FEATURES,
    extract_fft_features,
    parse_tls_payload_values,
    lag1_autocorr,
)

__all__ = [
    "FEATURE_LIST",
    "METADATA_FEATURES",
    "extract_fft_features",
    "parse_tls_payload_values",
    "lag1_autocorr",
    "convert_sessions_to_features",
]


def convert_sessions_to_features(session_iter, label=None):
    """Build the same feature DataFrame as tools.convert_pcap_to_features, but
    from an arbitrary iterator of TCPSession objects (e.g. a live extractor).
    """
    output = []
    session_counter = 0
    dropped_sessions = 0

    for session in session_iter:
        session_counter += 1
        if not session.duration:
            dropped_sessions += 1
            continue

        client_tuples = session.client_to_server_packets
        server_tuples = session.server_to_client_packets

        client_packets = len(client_tuples)
        server_packets = len(server_tuples)

        client_payloads = [p[1] for p in client_tuples]
        server_payloads = [p[1] for p in server_tuples]

        client_times = np.diff([p[0] for p in client_tuples])
        server_times = np.diff([p[0] for p in server_tuples])

        client_payload_lengths = [len(p) for p in client_payloads]
        server_payload_lengths = [len(p) for p in server_payloads]

        client_bytes = np.sum(client_payload_lengths)
        server_bytes = np.sum(server_payload_lengths)
        total_bytes = client_bytes + server_bytes

        if total_bytes == 0:
            dropped_sessions += 1
            continue

        combined_packets = sorted(client_tuples + server_tuples, key=lambda x: x[0])
        packet_sizes = np.array(
            [
                payload if isinstance(payload, (int, float)) else len(payload)
                for _, payload in combined_packets
            ],
            dtype=np.float32,
        )
        fft_features = extract_fft_features(packet_sizes)

        features_dict = {
            "client_ip": session.client_ip,
            "server_ip": session.server_ip,
            "port": session.client_port,
            "client_ratio": client_bytes / total_bytes,
            "server_ratio": server_bytes / total_bytes,
            "total_kbytes": total_bytes / 1024,
            "total_packets": client_packets + server_packets,
            "client_over_server": (client_bytes / server_bytes) if server_bytes > 0 else 0.0,
            "client_iat_std": float(np.std(client_times)) if len(client_times) > 0 else 0.0,
            "server_iat_std": float(np.std(server_times)) if len(server_times) > 0 else 0.0,
            "client_payload_length_coeffvar": (
                float(np.std(client_payload_lengths) / np.mean(client_payload_lengths))
                if np.mean(client_payload_lengths) > 0
                else 0.0
            ),
            "server_payload_length_coeffvar": (
                float(np.std(server_payload_lengths) / np.mean(server_payload_lengths))
                if np.mean(server_payload_lengths) > 0
                else 0.0
            ),
        }
        features_dict.update(fft_features)
        output.append(features_dict)

    df = DataFrame(output)
    if label is not None and not df.empty:
        df["label"] = label
    return df, {"sessions_seen": session_counter, "dropped": dropped_sessions}
