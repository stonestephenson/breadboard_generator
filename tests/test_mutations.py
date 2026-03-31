"""Unit tests for generator/mutations.py — mutation engine."""

import copy
import os
import json
from generator.mutations import MutationEngine

CIRCUIT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'config', 'circuits', 'simple_led.json'
)


def _load_circuit():
    with open(CIRCUIT_PATH) as f:
        return json.load(f)


class TestRemoveComponent:
    def test_removes_one_component(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        mutated, record = engine.remove_component(circuit)
        assert len(mutated['components']) == len(circuit['components']) - 1

    def test_record_has_removed_info(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        _, record = engine.remove_component(circuit)
        assert record['type'] == 'remove_component'
        assert 'component_id' in record
        assert 'removed' in record

    def test_specific_component(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        mutated, record = engine.remove_component(circuit, component_id='R1')
        assert record['component_id'] == 'R1'
        ids = [c['id'] for c in mutated['components']]
        assert 'R1' not in ids

    def test_original_unchanged(self):
        circuit = _load_circuit()
        original = copy.deepcopy(circuit)
        engine = MutationEngine(seed=1)
        engine.remove_component(circuit)
        assert circuit == original


class TestWrongPosition:
    def test_pins_shifted(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        mutated, record = engine.wrong_position(circuit)
        assert record['type'] == 'wrong_position'
        assert record['original_pins'] != record['mutated_pins']

    def test_shift_within_bounds(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        mutated, record = engine.wrong_position(circuit)
        for pin_val in record['mutated_pins'].values():
            assert 1 <= pin_val[1] <= 63

    def test_specific_component(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        _, record = engine.wrong_position(circuit, component_id='LED1')
        assert record['component_id'] == 'LED1'

    def test_original_unchanged(self):
        circuit = _load_circuit()
        original = copy.deepcopy(circuit)
        engine = MutationEngine(seed=1)
        engine.wrong_position(circuit)
        assert circuit == original


class TestWrongConnection:
    def test_wire_modified(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        mutated, record = engine.wrong_connection(circuit)
        assert record['type'] == 'wrong_connection'
        assert record['original'] != record['mutated']

    def test_specific_wire(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        _, record = engine.wrong_connection(circuit, wire_index=0)
        assert record['wire_index'] == 0

    def test_column_within_bounds(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        _, record = engine.wrong_connection(circuit)
        assert 1 <= record['mutated'][1] <= 63

    def test_original_unchanged(self):
        circuit = _load_circuit()
        original = copy.deepcopy(circuit)
        engine = MutationEngine(seed=1)
        engine.wrong_connection(circuit)
        assert circuit == original


class TestSwapPolarity:
    def test_rail_swapped(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        mutated, record = engine.swap_polarity(circuit)
        assert record['type'] == 'swap_polarity'
        swap = record['swaps'][0]
        assert swap['original'][0] != swap['mutated'][0]

    def test_plus_becomes_minus_or_vice_versa(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        _, record = engine.swap_polarity(circuit)
        swap = record['swaps'][0]
        orig_rail = swap['original'][0]
        mut_rail = swap['mutated'][0]
        # Should swap + <-> - on the same side
        assert orig_rail.replace('+', '-') == mut_rail or orig_rail.replace('-', '+') == mut_rail

    def test_original_unchanged(self):
        circuit = _load_circuit()
        original = copy.deepcopy(circuit)
        engine = MutationEngine(seed=1)
        engine.swap_polarity(circuit)
        assert circuit == original


class TestExtraComponent:
    def test_adds_one_component(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        mutated, record = engine.extra_component(circuit)
        assert len(mutated['components']) == len(circuit['components']) + 1

    def test_record_has_added_info(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        _, record = engine.extra_component(circuit)
        assert record['type'] == 'extra_component'
        assert 'added' in record
        assert record['added']['type'] in ('resistor', 'led')

    def test_original_unchanged(self):
        circuit = _load_circuit()
        original = copy.deepcopy(circuit)
        engine = MutationEngine(seed=1)
        engine.extra_component(circuit)
        assert circuit == original


class TestCompoundMutation:
    def test_applies_multiple(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        mutated, record = engine.compound_mutation(circuit, n_mutations=2)
        assert record['type'] == 'compound_mutation'
        assert record['n_applied'] >= 1

    def test_differs_from_original(self):
        circuit = _load_circuit()
        engine = MutationEngine(seed=1)
        mutated, _ = engine.compound_mutation(circuit, n_mutations=2)
        assert mutated != circuit

    def test_original_unchanged(self):
        circuit = _load_circuit()
        original = copy.deepcopy(circuit)
        engine = MutationEngine(seed=1)
        engine.compound_mutation(circuit)
        assert circuit == original


class TestReproducibility:
    def test_same_seed_same_result(self):
        circuit = _load_circuit()
        engine1 = MutationEngine(seed=42)
        engine2 = MutationEngine(seed=42)
        m1, r1 = engine1.wrong_position(circuit)
        m2, r2 = engine2.wrong_position(circuit)
        assert m1 == m2
        assert r1 == r2

    def test_different_seed_different_result(self):
        circuit = _load_circuit()
        engine1 = MutationEngine(seed=1)
        engine2 = MutationEngine(seed=99)
        m1, _ = engine1.wrong_position(circuit)
        m2, _ = engine2.wrong_position(circuit)
        # Very likely different (not guaranteed but with these seeds they are)
        assert m1 != m2
