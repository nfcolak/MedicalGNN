"""Per-field profile of a graphs.jsonl, to find exactly where two copies of the graph file differ. No values are printed.

    python analysis/graph_profile.py <graphs.jsonl> profile.json        # profile one file (standard library only)
    python analysis/graph_profile.py --compare a.json b.json            # compare two profiles

For every field path (`graph.<key>`, `nodes[].<key>`, `edges[].<key>`) the profile records how many graphs have it, the
total bytes of its values, and an order-independent fingerprint over graphs of its values (the values of a field are
taken in node or edge order within each graph, together with the sample id). Two files whose profiles agree on a field
hold identical values of that field for every graph; a field present in one profile only is an added or removed field.
Output is structure and counts only (no patient values), so profiles can be exchanged.
"""
import hashlib
import json
import sys
from collections import defaultdict

SEPARATORS = (",", ":")


def _feed(profile, path, sample_id, text):
    digest = int.from_bytes(hashlib.blake2b((sample_id + "|" + text).encode(), digest_size=16).digest(), "big")
    entry = profile.setdefault(path, {"graphs": 0, "bytes": 0, "xor": 0})
    entry["graphs"] += 1
    entry["bytes"] += len(text)
    entry["xor"] ^= digest


def profile_graph(profile, graph):
    sample_id = graph["sample_id"]
    for key, value in graph.items():
        if key not in ("nodes", "edges"):
            _feed(profile, f"graph.{key}", sample_id, json.dumps(value, sort_keys=True, separators=SEPARATORS))
    for group in ("nodes", "edges"):
        columns = defaultdict(list)
        for index, item in enumerate(graph[group]):
            for key, value in item.items():
                columns[key].append((index, value))
        for key, items in columns.items():
            _feed(profile, f"{group}[].{key}", sample_id, json.dumps(items, sort_keys=True, separators=SEPARATORS))


def profile_file(path):
    profile, graphs = {}, 0
    with open(path, "rb") as handle:
        for line in handle:
            profile_graph(profile, json.loads(line))
            graphs += 1
    return {"graphs": graphs, "fields": {k: {**v, "xor": format(v["xor"], "032x")} for k, v in sorted(profile.items())}}


def compare(a, b):
    """Markdown lines saying, field by field, whether two profiles agree."""
    fa, fb = a["fields"], b["fields"]
    lines = [f"graphs: {a['graphs']} vs {b['graphs']}", ""]
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    same = [p for p in sorted(set(fa) & set(fb)) if fa[p] == fb[p]]
    differ = [p for p in sorted(set(fa) & set(fb)) if fa[p] != fb[p]]
    lines.append(f"fields in both with identical values: {len(same)}")
    lines.append(f"fields only in the first file ({len(only_a)}): " + ", ".join(f"{p} ({fa[p]['graphs']} graphs, {fa[p]['bytes']} bytes)" for p in only_a))
    lines.append(f"fields only in the second file ({len(only_b)}): " + ", ".join(f"{p} ({fb[p]['graphs']} graphs, {fb[p]['bytes']} bytes)" for p in only_b))
    lines.append(f"fields in both with different values ({len(differ)}):")
    for p in differ:
        lines.append(f"  {p}: graphs {fa[p]['graphs']} vs {fb[p]['graphs']}, bytes {fa[p]['bytes']} vs {fb[p]['bytes']}")
    return lines


def main(argv):
    if len(argv) == 4 and argv[1] == "--compare":
        a, b = (json.load(open(p)) for p in argv[2:4])
        print("\n".join(compare(a, b)))
        return 0
    if len(argv) == 3:
        result = profile_file(argv[1])
        json.dump(result, open(argv[2], "w"), indent=1)
        print(f"profiled {result['graphs']} graphs, {len(result['fields'])} field paths -> {argv[2]}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
