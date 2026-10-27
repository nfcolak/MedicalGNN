"""Order-independent fingerprint of a graphs.jsonl (same graphs in another order give the same value).
usage: python analysis/graph_fingerprint.py <graphs.jsonl> [digests.tsv]
Standard library only; reads the file once (about 4 to 5 minutes for 6.4 GB) and changes nothing.
Prints the graph count, bytes, ordered sha256 (= the file hash), the XOR of the per-line sha256 (order independent),
edges, fold counts, measurement+vital nodes and censored nodes. With a second argument it also writes one line per
graph: sample_id <tab> sha256 of that graph's line (about 8 MB), so two files can be compared graph by graph."""
import hashlib, json, sys, collections
path = sys.argv[1]
out = open(sys.argv[2], 'w') if len(sys.argv) > 2 else None
whole = hashlib.sha256(); xor = 0; n = 0; size = 0; edges = 0
folds = collections.Counter(); mv = 0; cens = 0
with open(path, 'rb') as handle:
    for line in handle:
        whole.update(line); size += len(line)
        digest = hashlib.sha256(line).digest()
        xor ^= int.from_bytes(digest, 'big')
        g = json.loads(line); n += 1
        edges += len(g['edges']); folds[g['split']] += 1
        for node in g['nodes']:
            if node['kind'] in ('measurement', 'vital'):
                mv += 1
                if node.get('censored'): cens += 1
        if out: out.write(g['sample_id'] + '\t' + digest.hex() + '\n')
print('graphs', n, 'bytes', size, 'edges', edges, 'folds', dict(folds))
print('ordered sha256', whole.hexdigest())
print('order-independent xor of line hashes', format(xor, '064x'))
print('measurement+vital nodes', mv, 'censored nodes', cens)
