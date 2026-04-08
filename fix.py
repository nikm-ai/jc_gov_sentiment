import re, json

with open('data/seed_data.json') as f:
    data = json.load(f)

for r in data:
    m = re.search(r'(\d{1,2})[.\-_](\d{1,2})[.\-_](\d{4})', r['date'])
    if m:
        r['date'] = f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"

seen = set()
deduped = []
for r in data:
    if r['filename'] not in seen:
        seen.add(r['filename'])
        deduped.append(r)

deduped.sort(key=lambda r: r['date'], reverse=True)

with open('data/seed_data.json', 'w') as f:
    json.dump(deduped, f, indent=2)

print(f'Done. {len(deduped)} meetings.')
for r in deduped:
    flags = len(r.get('risk_flags', []))
    flag_str = f"  ⚑ {flags}" if flags else ""
    print(f"  {r['date']}  {r['signal']:8}  {r['score']:+.2f}{flag_str}  — {r['filename']}")
