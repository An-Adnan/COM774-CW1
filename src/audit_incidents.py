import csv
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/incident_event_log.csv'


def main():
    with SOURCE.open(newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        rows = list(reader)

    def show(message=''):
        print(message)


    show('INCIDENT DATA AUDIT')
    show(f'Events: {len(rows):,}')
    show(f'Columns: {len(columns)}')
    show(f'Incidents: {len(set(row["number"] for row in rows)):,}')
    unique_rows = {tuple(row[column] for column in columns) for row in rows}
    show(f'Exact duplicate rows: {len(rows) - len(unique_rows):,}')
    show('\nMissing values (blank or ?):')
    for column in columns:
        count = sum(row[column].strip() in ('', '?') for row in rows)
        if count:
            show(f'  {column}: {count:,} ({count / len(rows):.1%})')

    show('\nDate checks (day/month/year):')
    date_columns = ['opened_at', 'sys_created_at', 'sys_updated_at',
                    'resolved_at', 'closed_at']
    for column in date_columns:
        dates = []
        invalid = 0
        for row in rows:
            value = row[column].strip()
            if value in ('', '?'):
                continue
            try:
                dates.append(datetime.strptime(value, '%d/%m/%Y %H:%M'))
            except ValueError:
                invalid += 1
        show(f'  {column}: {invalid} invalid dates')
        if dates:
            show(f'    {min(dates):%Y-%m-%d} to {max(dates):%Y-%m-%d}')

    # Select the earliest observed event, not the final state of the incident.
    ordered = sorted(rows, key=lambda row: (
        datetime.strptime(row['sys_updated_at'], '%d/%m/%Y %H:%M'),
        int(row['sys_mod_count'])))
    first_events = {}
    for row in ordered:
        first_events.setdefault(row['number'], row)

    show('\nEarliest observed event states:')
    for state, count in Counter(row['incident_state'] for row in first_events.values()).most_common():
        show(f'  {state}: {count:,}')

    initial = [row for row in first_events.values()
               if row['incident_state'] == 'New'
               and int(row['sys_mod_count']) == 0
               and int(row['reassignment_count']) == 0
               and int(row['reopen_count']) == 0]
    show(f'\nCandidate initial records: {len(initial):,}')
    show('This is an audit selection; no source records were removed.')
    for target in ['priority', 'assignment_group']:
        counts = Counter(row[target] for row in initial
                         if row[target].strip() not in ('', '?'))
        missing = len(initial) - sum(counts.values())
        show(f'\n{target}: {len(counts)} classes; {missing} missing labels')
        for label, count in counts.most_common(5):
            show(f'  {label}: {count:,}')
        if len(counts) > 5:
            show(f'  Groups with fewer than 20 records: {sum(n < 20 for n in counts.values())}')

    show('\nCautions: repeated incident IDs are event history, not duplicates.')
    show('Earliest recorded events may occur after opening.')
    show('Exclude later resolution/closure information from early predictors.')
    show('Priority from impact/urgency needs a rule baseline.')
    show('\nNo cleaning, exclusions or model training performed.')
    show(f'SHA-256: {hashlib.sha256(SOURCE.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()

