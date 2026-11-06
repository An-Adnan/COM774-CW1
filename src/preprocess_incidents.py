import hashlib
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/incidents_processed.csv'
OUTPUT = SOURCE.parent
FEATURES = ['category', 'subcategory', 'u_symptom', 'contact_type', 'location']
TARGETS = ['priority', 'assignment_group']


def split_data(data):
    """Split once by record availability time; keep tied timestamps together."""
    data = data.sort_values(['sys_updated_at', 'number']).reset_index(drop=True)
    validation_start = data.iloc[int(len(data) * 0.70)]['sys_updated_at']
    test_start = data.iloc[int(len(data) * 0.85)]['sys_updated_at']
    train = data[data['sys_updated_at'] < validation_start].copy()
    validation = data[(data['sys_updated_at'] >= validation_start)
                      & (data['sys_updated_at'] < test_start)].copy()
    test = data[data['sys_updated_at'] >= test_start].copy()
    assert all(len(part) for part in [train, validation, test])
    assert train['sys_updated_at'].max() < validation['sys_updated_at'].min()
    assert validation['sys_updated_at'].max() < test['sys_updated_at'].min()
    assert len(train) + len(validation) + len(test) == len(data)
    assert set(train['number']).isdisjoint(validation['number'])
    assert set(train['number']).isdisjoint(test['number'])
    assert set(validation['number']).isdisjoint(test['number'])
    return {'train': train, 'validation': validation, 'test': test}


def prepare_target(partitions, target):
    """Keep known target labels; fit one pipeline on this task's training rows."""
    eligible = {name: part.dropna(subset=[target]).copy()
                for name, part in partitions.items()}
    if eligible['train'].empty:
        raise ValueError(f'No labelled training records for {target}.')
    preprocessor = Pipeline([
        ('missing', SimpleImputer(strategy='constant', fill_value='Unknown',
                                  keep_empty_features=True)),
        ('encoding', OneHotEncoder(handle_unknown='ignore'))
    ])
    # The five original fields are the entire predictor whitelist.
    encoded = {'train': preprocessor.fit_transform(eligible['train'][FEATURES])}
    for name in ['validation', 'test']:
        if eligible[name].empty:
            raise ValueError(f'No labelled {name} records for {target}.')
        encoded[name] = preprocessor.transform(eligible[name][FEATURES])
    labels = {name: part[target].copy() for name, part in eligible.items()}
    support = labels['train'].value_counts()
    # Balanced weights: training size / (number of classes * class count).
    weights = {label: len(labels['train']) / (len(support) * count)
               for label, count in support.items()}
    for name in encoded:
        assert encoded[name].shape[0] == len(labels[name])
        assert encoded[name].shape[1] == encoded['train'].shape[1]
        assert pd.notna(encoded[name].data).all()
    return {'preprocessor': preprocessor, 'X': encoded, 'y': labels,
            'ids': {name: part['number'].copy() for name, part in eligible.items()},
            'class_weights': weights, 'training_support': support}


def main(source=SOURCE, output=OUTPUT):
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    data = pd.read_csv(source, dtype=str)
    if data['number'].isna().any() or data['number'].duplicated().any():
        raise ValueError('Incident IDs must be present and unique.')
    data['sys_updated_at'] = pd.to_datetime(data['sys_updated_at'],
                                            format='%Y-%m-%d %H:%M:%S', errors='raise')
    if data['sys_updated_at'].isna().any():
        raise ValueError('Every record needs an update timestamp for splitting.')
    partitions = split_data(data)
    print('PREPROCESSING (CSV unchanged)')
    print('Inputs:', ', '.join(FEATURES))
    print('Chronological partitions, shared by both tasks:')
    for name, part in partitions.items():
        dates = part['sys_updated_at']
        print(f'  {name}: {len(part):,} incidents ({dates.min():%d %b %Y} to {dates.max():%d %b %Y})')

    tasks = {}
    for target in TARGETS:
        task = prepare_target(partitions, target)
        tasks[target] = task
        print(f'\n{target}:')
        for name, part in partitions.items():
            omitted = len(part) - len(task['y'][name])
            print(f'  {name}: {len(task["y"][name]):,} usable; {omitted} missing labels excluded')
        support = task['training_support']
        print(f'  Training classes: {len(support)}; encoded columns: {task["X"]["train"].shape[1]}')
        print(f'  Training classes below 20 examples: {sum(support < 20)} (labels retained)')
        for name in ['validation', 'test']:
            unseen = ~task['y'][name].isin(support.index)
            print(f'  {name} labels absent from training: {unseen.sum()} records')
        if target == 'priority':
            print('  Training priority support:')
            for label, count in support.items():
                print(f'    {label}: {count:,}; weight {task["class_weights"][label]:.2f}')
    print('\nMissing predictors become Unknown; encoders fit on training only.')
    print('Unseen input categories are accepted with an all-zero category block.')
    print('Class weights calculated from training labels, ready for CW2; not applied yet.')
    print('No labels merged, resampling, new features or classifier fitting.')
    print('No claim of useful predictive performance or optimal historical labels.')
    # Readable partitions retain all records and original columns.
    # Fill predictor blanks only; missing target labels stay blank.
    for name, part in partitions.items():
        readable = part.copy()
        readable[FEATURES] = readable[FEATURES].fillna('Unknown')
        readable.to_csv(output / f'{name}.csv', index=False,
                        date_format='%Y-%m-%d %H:%M:%S')

    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    print('Saved train.csv, validation.csv and test.csv (predictors filled; targets unchanged).')
    print('Encoding and class weights are recreated in memory; no joblib files saved.')
    return tasks


if __name__ == '__main__':
    main()
