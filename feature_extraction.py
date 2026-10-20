
from pandas import concat
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tools import convert_pcap_to_features, FEATURE_LIST
from debug_tools import convert_pcap_to_features_debug


def process_pcap_file(infile, debug, label):
    print(f"Processing {infile}...")
    if debug:
        out_df = convert_pcap_to_features_debug(infile)
    else:
        out_df = convert_pcap_to_features(infile, label)
        if not out_df.empty:
            out_df = out_df[FEATURE_LIST].reset_index(drop=True)
            out_df['label'] = label
    print(f"Finished processing {infile}. Extracted {len(out_df)} rows.")
    return out_df

def main():
    parser = ArgumentParser(description="Extract features from DNS PCAP")
    parser.add_argument("-i", "--input", required=True, help="Input PCAP file or directory")
    parser.add_argument("-o", "--output", required=True, help="Name for output csv")
    parser.add_argument("-l", "--label", type=str, help="Label to attach to output values (required unless --debug is used)")
    parser.add_argument("-d", "--debug", action="store_true", help="Provide debug output on feature extraction")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), help="Number of worker processes for directory mode")
    args = parser.parse_args()

    if args.debug:
        args.label = None
    elif args.label is None:
        parser.error("the following argument is required unless --debug is used: -l/--label")

    if os.path.isfile(args.input):
        if args.debug:
            output = convert_pcap_to_features_debug(args.input)
        else:
            output = convert_pcap_to_features(args.input, args.label)
            if not output.empty:
                output = output[FEATURE_LIST].reset_index(drop=True)
                output['label'] = args.label
        output.to_csv(args.output, index=False)
        

    elif os.path.isdir(args.input):
        pcap_files = [
            os.path.join(args.input, filename)
            for filename in os.listdir(args.input)
            if filename.endswith(('.pcap', '.pcapng'))
        ]

        all_dfs = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_pcap_file, infile, args.debug, args.label) for infile in pcap_files]
            for future in as_completed(futures):
                all_dfs.append(future.result())

        all_dfs = concat(all_dfs, ignore_index=True)
        all_dfs.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
