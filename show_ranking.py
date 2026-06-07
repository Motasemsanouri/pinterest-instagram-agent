import json, glob, os

files = glob.glob("data/results_*.json")
latest = max(files, key=os.path.getmtime)
print("FILE:", latest, "\n")

with open(latest, encoding="utf-8") as f:
    d = json.load(f)

pins = d["result"]["pins"]
for p in pins:
    print(f"#{p.get('rank_position')} pin={p['pin_id']} "
          f"score={p.get('rank_score')} {p.get('rank_breakdown')}")

print("\nranking_result:", json.dumps(d["result"].get("ranking_result"),
                                      ensure_ascii=False, indent=2))