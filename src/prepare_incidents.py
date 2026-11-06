import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/noisy/incident_events_noisy.csv'
OUTPUT = ROOT / 'data/processed/incidents_processed.csv'
EXPECTED_COLUMNS = '''number incident_state active reassignment_count reopen_count
sys_mod_count made_sla caller_id opened_by opened_at sys_created_by sys_created_at
sys_updated_by sys_updated_at contact_type location category subcategory u_symptom
cmdb_ci impact urgency priority assignment_group assigned_to knowledge
u_priority_confirmation notify problem_id rfc vendor caused_by closed_code
resolved_by resolved_at closed_at'''.split()
DATE_COLUMNS = ['opened_at', 'sys_created_at', 'sys_updated_at', 'resolved_at', 'closed_at']
COUNT_COLUMNS = ['reassignment_count', 'reopen_count', 'sys_mod_count']
BOOL_COLUMNS = ['active', 'made_sla', 'knowledge', 'u_priority_confirmation']
# Keep timestamps and impact/urgency for inspection, not automatic model inputs.
KEEP_COLUMNS = ['number', 'opened_at', 'sys_created_at', 'sys_updated_at',
                'contact_type', 'location', 'category', 'subcategory', 'u_symptom',
                'impact', 'urgency', 'priority', 'assignment_group']


def main(source=SOURCE, output=OUTPUT):
    source = Path(source)
    output = Path(output)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError('Unexpected CSV columns. Check the input before processing.')
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('Malformed CSV row found. Check the input before processing.')

    counts = Counter()
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = tuple(row[column] for column in EXPECTED_COLUMNS)
        if key in seen:
            counts['Exact duplicate events removed'] += 1
            continue
        seen.add(key)
        original_category = row['category']
        row = {column: value.strip() if value.strip() not in ('', '?') else None
               for column, value in row.items()}
        if not row['number']:
            raise ValueError('An event has no incident ID.')
        if row['category']:
            row['category'] = row['category'].capitalize()
            counts['Category formatting corrections'] += row['category'] != original_category
        for column in DATE_COLUMNS:
            if row[column] is None:
                continue
            value = row[column]
            for date_format in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M']:
                try:
                    row[column] = datetime.strptime(value, date_format)
                    if date_format == '%Y-%m-%d %H:%M':
                        counts['Alternative date formats standardised'] += 1
                    break
                except ValueError:
                    continue
            else:
                row[column] = None
                counts['Invalid dates set to missing'] += 1
        for column in COUNT_COLUMNS:
            try:
                value = int(row[column])
                if value < 0:
                    raise ValueError
                row[column] = value
            except (ValueError, TypeError):
                row[column] = None
                counts['Invalid counts set to missing'] += 1
        for column in BOOL_COLUMNS:
            if row[column] in ('true', 'false'):
                row[column] = row[column] == 'true'
            elif row[column] is not None:
                row[column] = None
                counts['Unexpected booleans set to missing'] += 1
        counts['Unknown state -100 events'] += row['incident_state'] == '-100'
        groups[row['number']].append(row)

    processed = []
    exclusions = Counter()
    for incident, events in groups.items():
        # Do not substitute a later event when the earliest event is uncertain.
        if any(r['sys_updated_at'] is None or r['sys_mod_count'] is None for r in events):
            exclusions['Uncertain event ordering'] += 1
            continue
        events.sort(key=lambda r: (r['sys_updated_at'], r['sys_mod_count']))
        first = events[0]
        if len(events) > 1 and (first['sys_updated_at'], first['sys_mod_count']) == (events[1]['sys_updated_at'], events[1]['sys_mod_count']):
            exclusions['Ambiguous earliest event'] += 1
            continue
        if not (first['incident_state'] == 'New' and first['sys_mod_count'] == 0
                and first['reassignment_count'] == 0 and first['reopen_count'] == 0):
            exclusions['No qualifying initial record'] += 1
            continue
        if first['opened_at'] is not None and first['sys_updated_at'] < first['opened_at']:
            exclusions['Initial update precedes opening'] += 1
            continue
        processed.append({column: first[column] for column in KEEP_COLUMNS})

    # Validate the final table; missing targets remain missing.
    assert len({row['number'] for row in processed}) == len(processed)
    assert len(processed) + sum(exclusions.values()) == len(groups)
    allowed_priority = {'1 - Critical', '2 - High', '3 - Moderate', '4 - Low', None}
    assert all(row['priority'] in allowed_priority for row in processed)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        for row in processed:
            writer.writerow({column: value.strftime('%Y-%m-%d %H:%M:%S')
                             if isinstance(value, datetime) else value
                             for column, value in row.items()})

    print('CLEANING AND PREPARATION')
    print(f'Input events: {len(rows):,}')
    for name, count in counts.items():
        print(f'{name}: {count:,}')
    print(f'Unique incidents: {len(groups):,}')
    for reason, count in exclusions.items():
        print(f'Excluded - {reason}: {count:,}')
    print(f'Processed incidents: {len(processed):,}')
    print(f'Output columns: {len(KEEP_COLUMNS)}')
    print('\nMissing values in processed table:')
    for column in KEEP_COLUMNS:
        count = sum(row[column] is None for row in processed)
        if count:
            print(f'  {column}: {count:,}')
    for target in ['priority', 'assignment_group']:
        values = Counter(row[target] for row in processed if row[target] is not None)
        print(f'\n{target}: {len(values)} classes. Most frequent:')
        for label, count in values.most_common(5):
            print(f'  {label}: {count:,}')
    print('\nMissing labels were not filled; use separate eligibility for each target.')
    print('Missing dates were not guessed. Dates have no supplied timezone.')
    print('Final outcomes, staff IDs, sparse fields and event counters are excluded from output.')
    print('Impact/urgency are inspection fields: exclude from the proposed priority ML inputs.')
    print('No new features, imputation, encoding, splitting or model training.')
    print('This cleaned CSV is the master dataset for subsequent preparation and modelling.')
    print('Input SHA-256:', source_hash)
    print('Output SHA-256:', hashlib.sha256(output.read_bytes()).hexdigest())
    print('Saved:', output)


if __name__ == '__main__':
    main()

