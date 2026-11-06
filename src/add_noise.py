import csv
import hashlib
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/incident_event_log.csv'
OUTPUT = ROOT / 'data/noisy/incident_events_noisy.csv'


def main(source=SOURCE, output=OUTPUT):
    source = Path(source)
    output = Path(output)
    with source.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)

    rng = random.Random(774)
    original_count = len(rows)
    error_count = round(original_count * 0.01)
    duplicate_count = round(original_count * 0.005)
    eligible = [i for i, row in enumerate(rows) if row['category'] not in ('', '?')]
    for index in rng.sample(eligible, error_count):
        rows[index]['category'] = '  ' + rows[index]['category'].upper() + '  '
    for index in rng.sample(range(original_count), error_count):
        value = datetime.strptime(rows[index]['opened_at'], '%d/%m/%Y %H:%M')
        rows[index]['opened_at'] = value.strftime('%Y-%m-%d %H:%M')
    for index in rng.sample(range(original_count), duplicate_count):
        rows.append(rows[index].copy())

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Category formatting errors: {error_count:,}')
    print(f'Alternative date formats: {error_count:,}')
    print(f'Added duplicates: {duplicate_count:,}')
    print(f'Output events: {len(rows):,}')
    print('Raw data and targets unchanged. Noise selections may overlap.')
    print('Source SHA-256:', hashlib.sha256(source.read_bytes()).hexdigest())
    print('Noisy SHA-256:', hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()

