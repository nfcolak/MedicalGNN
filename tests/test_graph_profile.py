"""analysis/graph_profile.py: where two copies of a graph file differ."""
import json

from analysis import graph_profile as gp


def graph(sid, *, extra=False, value=1.0, reverse_nodes=False):
    nodes = [{"id": "a", "kind": "patient", "token": "patient"}, {"id": "b", "kind": "measurement", "token": "lab:1", "value": value}]
    if extra:
        for n in nodes:
            n["availability_assumed"] = True
    if reverse_nodes:
        nodes = nodes[::-1]
    g = {"sample_id": sid, "split": "train", "nodes": nodes,
         "edges": [{"source": "a", "target": "b", "relation": "measured_in", "informative": False}]}
    if extra:
        g["temporal_validation"] = {"checked": True}
    return g


def write(path, graphs):
    path.write_text("".join(json.dumps(g) + "\n" for g in graphs))
    return gp.profile_file(path)


def test_same_graphs_in_another_order_have_identical_profiles(tmp_path):
    graphs = [graph(f"s{i}", value=i) for i in range(5)]
    a, b = write(tmp_path / "a.jsonl", graphs), write(tmp_path / "b.jsonl", graphs[::-1])
    assert a == b and gp.compare(a, b)[2] == f"fields in both with identical values: {len(a['fields'])}"


def test_added_fields_are_named_and_the_other_fields_still_agree(tmp_path):
    a = write(tmp_path / "a.jsonl", [graph(f"s{i}") for i in range(4)])
    b = write(tmp_path / "b.jsonl", [graph(f"s{i}", extra=True) for i in range(4)])
    text = "\n".join(gp.compare(a, b))
    assert "temporal_validation" in text and "nodes[].availability_assumed" in text
    assert "fields in both with different values (0)" in text
    assert "fields only in the first file (0)" in text


def test_a_changed_value_and_a_changed_node_order_are_detected(tmp_path):
    base = [graph(f"s{i}", value=i) for i in range(4)]
    changed = write(tmp_path / "c.jsonl", [graph("s0", value=99)] + base[1:])
    a = write(tmp_path / "a.jsonl", base)
    text = "\n".join(gp.compare(a, changed))
    assert f"fields in both with identical values: {len(a['fields']) - 1}" in text
    assert "nodes[].value" in text and "(1)" in text.split("fields in both with different values")[1].split("\n")[0]
    reordered = write(tmp_path / "r.jsonl", [graph("s0", value=0, reverse_nodes=True)] + base[1:])
    text = "\n".join(gp.compare(a, reordered))
    assert "fields in both with different values (0)" not in text and "nodes[]" in text.split("different values")[1]


def test_nothing_but_structure_is_in_a_profile(tmp_path):
    p = write(tmp_path / "a.jsonl", [graph("secret-id", value=123456.789)])
    assert "123456.789" not in json.dumps(p) and "secret-id" not in json.dumps(p)


def test_values_swapped_between_two_graphs_are_not_mistaken_for_the_same_file(tmp_path):
    a = write(tmp_path / "a.jsonl", [graph("s0", value=1), graph("s1", value=2)])
    b = write(tmp_path / "b.jsonl", [graph("s0", value=2), graph("s1", value=1)])
    assert "nodes[].value" in "\n".join(gp.compare(a, b)).split("different values")[1]
