import json
import os

RECORDS_FILE = os.path.join("levels", "records.json")

def load_records():
    if not os.path.exists(RECORDS_FILE):
        return {}
    with open(RECORDS_FILE, 'r') as f:
        try:
            return json.load(f)
        except:
            return {}

def save_records(records):
    os.makedirs("levels", exist_ok=True)
    with open(RECORDS_FILE, 'w') as f:
        json.dump(records, f, indent=2)

def get_pb(level_filename):
    """Возвращает (time, deaths) или None, если рекорда нет."""
    records = load_records()
    entry = records.get(level_filename)
    if entry:
        return (entry['time'], entry.get('deaths', 0))
    return None

def save_record(level_filename, time, deaths):
    """Сохраняет рекорд, если он лучше предыдущего."""
    records = load_records()
    current = records.get(level_filename)
    if current is None or time < current['time']:
        records[level_filename] = {'time': time, 'deaths': deaths}
        save_records(records)