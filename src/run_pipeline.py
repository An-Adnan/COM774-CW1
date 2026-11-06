"""Run the existing CW1 pipeline from a raw CSV, locally or in Azure."""
import argparse
import hashlib
from pathlib import Path

import add_noise
import prepare_incidents
import preprocess_incidents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Raw incident CSV')
    parser.add_argument('--output', type=Path, required=True, help='Output folder')
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise ValueError('Input must be a single raw CSV file, not a folder or MLTable.')
    if source.is_relative_to(output):
        raise ValueError('Keep the input file outside the output folder.')

    # Check the input before adding synthetic noise.
    import csv
    with source.open(newline='', encoding='utf-8-sig') as file:
        if csv.DictReader(file).fieldnames != prepare_incidents.EXPECTED_COLUMNS:
            raise ValueError('Expected the original raw incident event log (36 columns).')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    print('RAW INPUT SHA-256:', source_hash)
    noisy = output / 'noisy/incident_events_noisy.csv'
    processed = output / 'processed/incidents_processed.csv'
    add_noise.main(source, noisy)
    prepare_incidents.main(noisy, processed)
    preprocess_incidents.main(processed, processed.parent)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    for path in sorted(output.rglob('*.csv')):
        print(path.relative_to(output), 'SHA-256:',
              hashlib.sha256(path.read_bytes()).hexdigest())
    print('Pipeline complete. Raw input unchanged. No model trained.')


if __name__ == '__main__':
    main()
