from pandas import DataFrame
import numpy as np
from tcp_sessions_ext import TCPSessionExtractor
import tensorflow as tf
from scipy.integrate import simpson

# Baseline features
FEATURE_LIST = ['client_ratio', 'server_ratio', 'fft_diff_abs_mean', 'fft_low_band_power_ratio', 'server_first_packet']

# FEATURE_LIST = ['client_ratio', 'server_ratio', 'server_first_packet',
#                 'fft_diff_abs_mean', 'fft_low_band_power_ratio', 'fft_peak_power_ratio']


METADATA_FEATURES = ['client_ip','server_ip','port']


def extract_fft_features(packet_sizes: np.ndarray, prefix: str = 'fft_'):
    n = len(packet_sizes)
    default_features = {
        f'{prefix}spectral_entropy': 0.0,
        f'{prefix}low_band_power_ratio': 0.0,
        f'{prefix}series_slope': 0.0,
        f'{prefix}half_energy_ratio': 0.0,
        f'{prefix}half_energy_contrast': 0.0,
        f'{prefix}diff_abs_mean': 0.0,
        f'{prefix}diff_energy': 0.0,
        f'{prefix}diff_zero_cross_rate': 0.0,
        f'{prefix}diff_pos_neg_energy_ratio': 0.0,
        f'{prefix}fall_time': 0.0,
        f'{prefix}85_power_bandwidth': 0.0,
        f'{prefix}spectral_centroid': 0.0,
        f'{prefix}peak_power_ratio': 0.0,
        f'{prefix}spectral_flatness': 0.0,
        f'{prefix}spectral_skew': 0.0,
        f'{prefix}spectral_kurtosis': 0.0,
        f'{prefix}dominant_frequency': 0.0,
    }

    if n <= 1:
        return default_features

    x = np.arange(n, dtype=np.float64)
    y = packet_sizes.astype(np.float64)
    series_slope = float(np.polyfit(x, y, 1)[0])

    diffs = np.diff(y)
    diff_abs_mean = float(np.mean(np.abs(diffs)))
    diff_energy = float(np.sum(np.square(diffs)))
    positive_diff_energy = float(np.sum(np.square(diffs[diffs > 0])))
    negative_diff_energy = float(np.sum(np.square(diffs[diffs < 0])))
    eps = 1e-10
    diff_pos_neg_energy_ratio = positive_diff_energy / (negative_diff_energy + eps)

    # Count sign changes between adjacent non-zero derivative samples.
    non_zero_signs = np.sign(diffs[diffs != 0])
    if len(non_zero_signs) > 1:
        sign_changes = np.sum(non_zero_signs[1:] != non_zero_signs[:-1])
        diff_zero_cross_rate = float(sign_changes / (len(non_zero_signs) - 1))
    else:
        diff_zero_cross_rate = 0.0

    # Time-domain half-energy descriptors over the sequence fed into the FFT.
    split_idx = n // 2
    first_half = y[:split_idx]
    second_half = y[split_idx:]
    e1 = float(np.sum(np.square(first_half)))
    e2 = float(np.sum(np.square(second_half)))
    half_energy_ratio = e1 / (e2 + eps)
    half_energy_contrast = (e1 - e2) / (e1 + e2 + eps)

    fft_result = tf.signal.rfft(tf.convert_to_tensor(packet_sizes, dtype=tf.float32))
    magnitudes = tf.abs(fft_result).numpy()
    powers = np.square(magnitudes)

    if len(magnitudes) == 0:
        return default_features

    non_dc_magnitudes = magnitudes[1:]
    non_dc_powers = powers[1:]
    non_dc_bins = np.arange(1, len(magnitudes), dtype=np.int32)

    if len(non_dc_magnitudes) == 0:
        return default_features

    total_non_dc_power = float(np.sum(non_dc_powers))
    
    # Spectral Entropy
    spectral_entropy = 0.0
    if total_non_dc_power > 0:
        p = non_dc_powers / total_non_dc_power
        p = p[p > 0]
        if len(p) > 1:
            spectral_entropy = float(-np.sum(p * np.log2(p)) / np.log2(len(non_dc_powers)))

    # Low-frequency energy ratio (lower third of frequency spectrum)
    low_band, _, _ = np.array_split(non_dc_powers, 3)
    low_band_power_ratio = float(np.sum(low_band) / total_non_dc_power) if total_non_dc_power > 0 else 0.0

    # 85% Power Bandwidth (frequency bin where 85% of cumulative power is reached)
    power_85_bin = 0.0
    if total_non_dc_power > 0:
        cumsum_power = np.cumsum(non_dc_powers)
        rolloff_idx = int(np.searchsorted(cumsum_power, 0.85 * total_non_dc_power, side='left'))
        rolloff_idx = min(rolloff_idx, len(non_dc_bins) - 1)
        power_85_bin = float(non_dc_bins[rolloff_idx])

    # Spectral Centroid (center of mass of the spectrum)
    spectral_centroid = 0.0
    if total_non_dc_power > 0:
        spectral_centroid = float(np.sum(non_dc_bins * non_dc_powers) / total_non_dc_power)

    # Peak Power Ratio (concentration of power at dominant frequency)
    peak_power_ratio = 0.0
    if total_non_dc_power > 0:
        peak_power_ratio = float(np.max(non_dc_powers) / total_non_dc_power)

    # Spectral Flatness (ratio of geometric to arithmetic mean)
    spectral_flatness = 0.0
    if len(non_dc_magnitudes) > 0 and total_non_dc_power > 0:
        geometric_mean = float(np.exp(np.mean(np.log(non_dc_magnitudes + 1e-10))))
        arithmetic_mean = float(np.mean(non_dc_magnitudes))
        if arithmetic_mean > 0:
            spectral_flatness = geometric_mean / arithmetic_mean

    # Spectral skew/kurtosis: 3rd/4th standardized moments of the power spectrum
    # about the spectral centroid (treating normalized power as a probability mass).
    spectral_skew = 0.0
    spectral_kurtosis = 0.0
    if total_non_dc_power > 0:
        p = non_dc_powers / total_non_dc_power
        deviation = non_dc_bins - spectral_centroid
        variance = float(np.sum(p * np.square(deviation)))
        if variance > 0:
            std = np.sqrt(variance)
            spectral_skew = float(np.sum(p * np.power(deviation / std, 3)))
            spectral_kurtosis = float(np.sum(p * np.power(deviation / std, 4)))

    # Dominant frequency bin (peak of non-DC spectrum).
    dominant_frequency = float(non_dc_bins[int(np.argmax(non_dc_powers))])

    # Fall time in frequency bins from 90% to 10% of dominant non-DC magnitude.
    fall_time = 0.0
    peak_idx = int(np.argmax(non_dc_magnitudes))
    peak_mag = float(non_dc_magnitudes[peak_idx])
    if peak_mag > 0 and peak_idx < len(non_dc_magnitudes) - 1:
        trailing = non_dc_magnitudes[peak_idx:]
        below_90 = np.where(trailing <= 0.9 * peak_mag)[0]
        below_10 = np.where(trailing <= 0.1 * peak_mag)[0]
        if len(below_90) > 0 and len(below_10) > 0:
            idx_90 = peak_idx + int(below_90[0])
            idx_10 = peak_idx + int(below_10[0])
            if idx_10 >= idx_90:
                fall_time = float(non_dc_bins[idx_10] - non_dc_bins[idx_90])

    # commenting out uninformative metrics based on CDF appearance and stats
    return {
        #f'{prefix}spectral_entropy': spectral_entropy,
        f'{prefix}low_band_power_ratio': low_band_power_ratio,
        # f'{prefix}series_slope': series_slope,
        # f'{prefix}half_energy_ratio': half_energy_ratio,
        # f'{prefix}half_energy_contrast': half_energy_contrast,
        f'{prefix}diff_abs_mean': diff_abs_mean,
        # f'{prefix}diff_energy': diff_energy,
        # f'{prefix}diff_zero_cross_rate': diff_zero_cross_rate,
        # f'{prefix}diff_pos_neg_energy_ratio': diff_pos_neg_energy_ratio,
        # f'{prefix}fall_time': fall_time,
        # f'{prefix}85_power_bandwidth': power_85_bin,
        #f'{prefix}spectral_centroid': spectral_centroid,
        f'{prefix}peak_power_ratio': peak_power_ratio,
        # f'{prefix}spectral_flatness': spectral_flatness,
        #f'{prefix}spectral_skew': spectral_skew,
        # f'{prefix}spectral_kurtosis': spectral_kurtosis,
        # f'{prefix}dominant_frequency': dominant_frequency,
    }

def parse_tls_payload_values(tls_payload: bytes):
    """
    Summarize the TLS records visible in a single packet by summing the
    record-layer content types.

    Rules:
    - Each complete TLS record contributes its content type once.
    - A truncated/partial TLS record at the end of the packet also contributes
      its content type once, as long as its full 5-byte TLS header is present.
    - If the payload stops before a full TLS header is available, parsing stops.
    - If a byte sequence does not look like a plausible TLS record header,
      return -1 and a debug structure.

    Returns:
        (summary, debug_dict)

        summary:
            int: Sum of TLS content types for all complete or partial records found.
                 Returns -1 for invalid / implausible TLS framing.

        debug_dict:
            dict with parsing details.
    """

    CT_CHANGE_CIPHER_SPEC = 20
    CT_ALERT = 21
    CT_HANDSHAKE = 22
    CT_APPLICATION_DATA = 23
    CT_HEARTBEAT = 24

    VALID_CT = {
        CT_CHANGE_CIPHER_SPEC,
        CT_ALERT,
        CT_HANDSHAKE,
        CT_APPLICATION_DATA,
        CT_HEARTBEAT,
    }

    CONTENT_TYPE_NAMES = {
        CT_CHANGE_CIPHER_SPEC: "change_cipher_spec",
        CT_ALERT: "alert",
        CT_HANDSHAKE: "handshake",
        CT_APPLICATION_DATA: "application_data",
        CT_HEARTBEAT: "heartbeat",
    }

    MAX_RECORD_LEN = 16384  # TLSPlaintext.fragment max

    n = len(tls_payload)
    i = 0
    summary = 0

    debug = {
        "payload_size": n,
        "records_seen": 0,
        "complete_records": 0,
        "partial_records": 0,
        "trailing_bytes_without_header": 0,
        "content_type_sum": 0,
        "content_type_counts": {
            "change_cipher_spec": 0,
            "alert": 0,
            "handshake": 0,
            "application_data": 0,
            "heartbeat": 0,
        },
        "records": [],
        "error": None,
    }

    while i < n:
        # Not enough bytes left for another TLS header.
        if i + 5 > n:
            debug["trailing_bytes_without_header"] = n - i
            break

        record_offset = i
        content_type = tls_payload[i]
        ver_major = tls_payload[i + 1]
        ver_minor = tls_payload[i + 2]
        length = int.from_bytes(tls_payload[i + 3:i + 5], byteorder="big")

        # Plausibility checks
        if content_type not in VALID_CT:
            debug["error"] = {
                "offset": record_offset,
                "reason": "invalid_content_type",
                "content_type": content_type,
            }
            return -1, debug

        if ver_major != 0x03 or ver_minor not in (0, 1, 2, 3, 4):
            debug["error"] = {
                "offset": record_offset,
                "reason": "invalid_tls_version",
                "version": (ver_major, ver_minor),
            }
            return -1, debug

        if length > (MAX_RECORD_LEN + 2048):
            debug["error"] = {
                "offset": record_offset,
                "reason": "record_length_too_large",
                "length": length,
            }
            return -1, debug

        ct_name = CONTENT_TYPE_NAMES[content_type]
        record_end = i + 5 + length
        is_complete = record_end <= n

        summary += content_type
        debug["records_seen"] += 1
        debug["content_type_sum"] = summary
        debug["content_type_counts"][ct_name] += 1

        record_info = {
            "offset": record_offset,
            "content_type": content_type,
            "content_type_name": ct_name,
            "version": f"{ver_major}.{ver_minor}",
            "declared_length": length,
            "available_body_bytes": max(0, n - (i + 5)),
            "is_complete": is_complete,
        }

        debug["records"].append(record_info)

        if is_complete:
            debug["complete_records"] += 1
            i = record_end
        else:
            debug["partial_records"] += 1
            break

    return summary, debug

def convert_pcap_to_features(infile, label=None):

    extractor = TCPSessionExtractor(
        pcap_path=infile,
        max_session_duration_sec=0,
        stale_session_timeout_sec=300,
        max_packets_per_session=5000,
        max_session_packets=30,
        show_progress=False,
        min_packets_per_direction=2,
    )
    output = []
    session_counter = 0
    dropped_sessions = 0

    for session in extractor:
        session_counter += 1
        duration = session.duration

        client_tuples= session.client_to_server_packets
        server_tuples = session.server_to_client_packets

        client_payloads = [packet[1] for packet in client_tuples]
        server_payloads = [packet[1] for packet in server_tuples]
        # get timestamps from tuples
        client_times = [packet[0] for packet in client_tuples]
        server_times = [packet[0] for packet in server_tuples]
        # get interarrival times
        client_times = np.diff(client_times)
        server_times = np.diff(server_times)
        client_payload_lengths = [len(p) for p in client_payloads]
        server_payload_lengths = [len(p) for p in server_payloads]
        client_packets = len(client_tuples)
        server_packets = len(server_tuples)

        # temp move of these features to evaluate clipped values
        client_bytes = np.sum(client_payload_lengths)
        server_bytes = np.sum(server_payload_lengths)
        total_bytes = client_bytes + server_bytes
        server_first_packet = server_payload_lengths[0]
            

        combined_packets = sorted(client_tuples + server_tuples, key=lambda x: x[0])

        packet_sizes = np.array([
            payload if isinstance(payload, (int, float)) else len(payload)
            for _, payload in combined_packets
        ], dtype=np.float32)
        combined_times = np.array([ts for ts, _ in combined_packets], dtype=np.float64)
        combined_interarrival = np.diff(combined_times).astype(np.float32)
        fft_features = extract_fft_features(packet_sizes)
        #fft_time_features = extract_fft_features(combined_interarrival, prefix='fft_time_')

        features_dict = {
            'client_ip': session.client_ip,
            'server_ip': session.server_ip,
            'port': session.client_port,
            'client_ratio': (client_bytes / total_bytes),
            'server_ratio':  (server_bytes / total_bytes),
            'server_first_packet': (server_first_packet / server_bytes),
            'client_packet_rate': client_packets / duration,

        }
        features_dict.update(fft_features)
        output.append(features_dict)
    output = DataFrame(output)
    print(f"Session generator statistics: {extractor.stats}")
    print (f"Total dropped sessions: {dropped_sessions} out of {session_counter} total sessions")
    output['label'] = label
    return DataFrame(output)


