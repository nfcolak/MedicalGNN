"""LCL-07: masked-node pre-training of the encoder + message-passing tensors of ``cei_gnn_mp2``.

Usage (nothing runs without ``--execute``)::

    python3 -m cei.pretrain --execute --v3-state STATE.json --k 4 --num-classes 10 \\
        --num-triples T --graphs graphs.pt --graph-ids ids.json \\
        --dev-ids dev.json --validation-ids val.json --test-ids test.json \\
        --train-ids train.json [--subject-map subjects.json] --out pretrain.pt \\
        [--method-option mp_mask_rate=0.15 ...]

``graphs.pt`` is a ``torch.save``d list of tensorized clinical graphs (the runner's format);
``ids.json`` is the aligned list of their graph ids. The objective is the network's own
``mask_recon_loss`` (LCL-02, ``mp_mask_rate > 0``); the heads get no gradient. The saved file is a
state dict of the tensors under ``cei.plugin_cei_gnn_mp2.PRETRAIN_PREFIXES`` and loads through
the ``mp_pretrain_checkpoint`` option; ``<out>.json`` beside it records the configuration the tensors
belong to (ple_mode, edge direction, width, MP kind) and the loader refuses a network that differs.
Any pre-training id that is also in the dev, validation, test or supervised train list stops the run
before the first step (all four lists are mandatory: by default the pre-training set must not hold the
10,000 supervised training graphs either). ``--allow-supervised-train`` (mp2rev-train-08) lifts only the
train-list check, because the masked-node objective uses no labels: dev, validation and test stay refused,
and the sidecar records ``supervised_train_ids_included`` and ``supervised_train_overlap``. Ids are graph ids; ``--subject-map`` (JSON {graph id: patient id}
covering every listed id) makes the same check on patients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

DEFAULT_MASK_RATE = 0.15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python3 -m cei.pretrain', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true',
                        help='actually run; without it the program only prints this usage and exits')
    parser.add_argument('--v3-state', help='cei_gnn_v3 state JSON (arm/k/vocabulary/knots)')
    parser.add_argument('--k', type=int, help='K of the state')
    parser.add_argument('--arm', default='C')
    parser.add_argument('--num-classes', type=int)
    parser.add_argument('--num-triples', type=int)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--token-dim', type=int, default=32)
    parser.add_argument('--edge-direction', default='forward', choices=('forward', 'bidirectional'))
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--graphs', help='torch file with the list of pre-training graphs')
    parser.add_argument('--graph-ids', help='JSON list of graph ids aligned with --graphs')
    parser.add_argument('--dev-ids', help='JSON list of dev graph ids (refused if overlapping)')
    parser.add_argument('--validation-ids', help='JSON list of validation graph ids (refused if overlapping)')
    parser.add_argument('--test-ids', help='JSON list of test graph ids (refused if overlapping)')
    parser.add_argument('--train-ids', help='JSON list of the supervised train ids (refused if overlapping)')
    parser.add_argument('--allow-supervised-train', action='store_true',
                        help='allow graphs of the supervised --train-ids list in the pre-training set (the '
                             'objective is label-free); dev/validation/test overlap is still refused; recorded '
                             'in the sidecar')
    parser.add_argument('--subject-map', help='optional JSON {graph id: patient id}; the overlap check then '
                                              'also runs on patients')
    parser.add_argument('--method-option', action='append', default=[], metavar='KEY=VALUE',
                        help='cei_gnn_mp2 option (repeatable); mp_mask_rate defaults to '
                             f'{DEFAULT_MASK_RATE}')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1.79e-3)
    parser.add_argument('--out', help='output checkpoint path (must not exist)')
    return parser


def _ids(path, label):
    values = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(values, list):
        raise SystemExit(f'--{label} must hold a JSON list')
    if not values:   # an empty list would make its leakage check vacuous
        raise SystemExit(f'--{label} must not be empty (all id lists are mandatory)')
    return [str(value) for value in values]


def _check_arguments(args, parser):
    required = ('v3_state', 'k', 'num_classes', 'num_triples', 'graphs', 'graph_ids', 'dev_ids',
                'validation_ids', 'test_ids', 'train_ids', 'out')
    missing = [f'--{name.replace("_", "-")}' for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f'--execute needs {", ".join(missing)}')
    if Path(args.out).exists():
        parser.error(f'--out {args.out!r} already exists; refusing to overwrite')
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be >= 1')
    ids = _ids(args.graph_ids, 'graph-ids')
    if len(set(ids)) != len(ids):
        parser.error('--graph-ids contains duplicate ids')
    forbidden = {'dev': args.dev_ids, 'validation': args.validation_ids, 'test': args.test_ids,
                 'train': args.train_ids}
    subject_of = None
    if args.subject_map is not None:
        document = json.loads(Path(args.subject_map).read_text(encoding='utf-8'))
        if not isinstance(document, dict):
            parser.error('--subject-map must hold a JSON object {graph id: patient id}')
        subject_of = {str(key): str(value) for key, value in document.items()}
    listed = {label: _ids(path, f'{label}-ids') for label, path in forbidden.items()}
    checked = {label: values for label, values in listed.items()
               if not (label == 'train' and args.allow_supervised_train)}
    for label, values in checked.items():
        overlap = sorted(set(ids).intersection(values))
        if overlap:
            raise SystemExit(f'refusing to pre-train: {len(overlap)} graph id(s) are in the {label} list '
                             f'(first: {overlap[:5]})')
    if subject_of is not None:
        unmapped = sorted({i for values in (ids, *listed.values()) for i in values if i not in subject_of})
        if unmapped:
            parser.error(f'--subject-map lacks {len(unmapped)} listed id(s) (first: {unmapped[:5]})')
        patients = {subject_of[i] for i in ids}
        for label, values in checked.items():
            shared = sorted(patients.intersection(subject_of[i] for i in values))
            if shared:
                raise SystemExit(f'refusing to pre-train: {len(shared)} patient(s) of the pre-training graphs '
                                 f'are in the {label} list (first: {shared[:5]})')
    return ids


def _method_options(pairs):
    options = {}
    for pair in pairs:
        key, separator, value = pair.partition('=')
        if not separator or not key.strip():
            raise SystemExit(f'--method-option {pair!r} is not KEY=VALUE')
        options[key.strip()] = value.strip()
    return options


def run(args, ids):
    import torch
    from torch_geometric.loader import DataLoader

    from .plugin_cei_gnn_mp2 import EvidenceAdapterMP2, checkpoint_config, checkpoint_state

    torch.manual_seed(args.seed)   # init and dropout draw from the global stream, like core.train
    graphs = torch.load(args.graphs, map_location='cpu', weights_only=False)
    if not isinstance(graphs, list) or len(graphs) != len(ids) or not graphs:
        raise SystemExit('--graphs must be a non-empty list aligned with --graph-ids')
    # --graph-ids is user-typed; the graphs carry their own sample_id, which must agree with it
    # (the dev/validation/test/train overlap check ran on --graph-ids)
    own = [(index, str(graph.sample_id)) for index, graph in enumerate(graphs)
           if getattr(graph, 'sample_id', None) is not None]
    mismatch = [(index, sample_id, ids[index]) for index, sample_id in own if sample_id != ids[index]]
    if mismatch:
        raise SystemExit(f'refusing to pre-train: {len(mismatch)} graph(s) carry a sample_id that differs from '
                         f'--graph-ids (first: position, sample_id, listed id = {mismatch[:3]})')
    from cei.plugin_cei_gnn_v3 import load_v3_state
    state = load_v3_state(args.v3_state)
    method_options = {'arm': args.arm, 'v3_state': args.v3_state, 'k': args.k,
                      'mp_mask_rate': DEFAULT_MASK_RATE, **_method_options(args.method_option)}
    if float(method_options['mp_mask_rate']) <= 0.0:
        raise SystemExit('pre-training needs mp_mask_rate > 0 (the masked-node objective)')
    namespace = Namespace(method_options=method_options, seed=args.seed, edge_direction=args.edge_direction)
    first = graphs[0]
    model = EvidenceAdapterMP2(
        num_tokens=state.num_tokens, node_dim=int(first.x.size(1)), edge_dim=int(first.edge_attr.size(1)),
        num_classes=args.num_classes, hidden=args.hidden, layers=args.layers, dropout=args.dropout,
        token_dim=args.token_dim, num_triples=args.num_triples, args=namespace)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True, generator=generator)
    model.train()
    history = []
    for epoch in range(args.epochs):
        total, seen = 0.0, 0
        for batch in loader:
            clinical, membership, visit_graph = model._read(batch)
            features = model.network.continuous_inputs(clinical)
            parts = model.network.forward_continuous(features, clinical.edge_index, clinical, membership,
                                                     return_parts=True, visit_graph=visit_graph)
            loss = parts['mask_recon_loss']
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * batch.num_graphs
            seen += batch.num_graphs
        history.append(round(total / seen, 6))
        print(json.dumps({'epoch': epoch + 1, 'mask_recon_loss': history[-1]}), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_state(model.network), out)
    meta = {'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'graphs': len(ids),
            'graph_ids_sha256': hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
            'epochs': args.epochs, 'seed': args.seed, 'loss_history': history,
            'supervised_train_ids_included': bool(args.allow_supervised_train),
            'supervised_train_overlap': len(set(ids).intersection(_ids(args.train_ids, 'train-ids'))),
            'config': checkpoint_config(model),
            'method_options': {key: str(value) for key, value in method_options.items()}}
    out.with_suffix(out.suffix + '.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'saved': str(out), 'sha256': meta['sha256']}))


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        parser.print_usage()
        print('nothing was run: pass --execute (and the id lists) to start masked-node pre-training',
              file=sys.stderr)
        return 2
    ids = _check_arguments(args, parser)
    run(args, ids)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
