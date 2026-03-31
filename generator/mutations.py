"""
Mutation engine — injects errors into valid circuit configs.

Each mutation takes a valid circuit config dict and returns a mutated
copy plus a mutation_record describing exactly what changed.
All randomness is seeded for reproducibility.
"""

import copy
import random
from typing import Optional


# Terminal rows and rail IDs for generating random positions
_TERMINAL_ROWS = list("abcdefghij")
_RAIL_IDS = ["p1+", "p1-", "p2-", "p2+"]


class MutationEngine:
    """Generates mutated (incorrect) variants of valid circuit configs."""

    def __init__(self, seed: int = 42, n_cols: int = 63):
        """
        Args:
            seed: Random seed for reproducibility.
            n_cols: Number of terminal columns on the board.
        """
        self.rng = random.Random(seed)
        self.n_cols = n_cols

    def remove_component(
        self, circuit: dict, component_id: Optional[str] = None,
    ) -> tuple[dict, dict]:
        """
        Remove a component from the circuit.

        Args:
            circuit: Valid circuit config dict.
            component_id: ID of component to remove, or None for random.

        Returns:
            (mutated_circuit, mutation_record)
        """
        mutated = copy.deepcopy(circuit)
        components = mutated['components']

        if not components:
            raise ValueError("Circuit has no components to remove")

        if component_id:
            idx = next(i for i, c in enumerate(components) if c['id'] == component_id)
        else:
            idx = self.rng.randrange(len(components))

        removed = components.pop(idx)

        record = {
            "type": "remove_component",
            "component_id": removed['id'],
            "removed": removed,
        }
        return mutated, record

    def wrong_position(
        self, circuit: dict, component_id: Optional[str] = None, max_shift: int = 5,
    ) -> tuple[dict, dict]:
        """
        Shift a component's pins to nearby wrong positions.

        Args:
            circuit: Valid circuit config dict.
            component_id: ID of component to shift, or None for random.
            max_shift: Maximum column shift (1 to max_shift).

        Returns:
            (mutated_circuit, mutation_record)
        """
        mutated = copy.deepcopy(circuit)
        components = mutated['components']

        if not components:
            raise ValueError("Circuit has no components to shift")

        if component_id:
            comp = next(c for c in components if c['id'] == component_id)
        else:
            comp = self.rng.choice(components)

        shift = self.rng.choice([s for s in range(-max_shift, max_shift + 1) if s != 0])
        original_pins = copy.deepcopy(comp['pins'])

        for pin_name, pin_val in comp['pins'].items():
            new_col = pin_val[1] + shift
            new_col = max(1, min(self.n_cols, new_col))
            pin_val[1] = new_col

        record = {
            "type": "wrong_position",
            "component_id": comp['id'],
            "shift": shift,
            "original_pins": original_pins,
            "mutated_pins": copy.deepcopy(comp['pins']),
        }
        return mutated, record

    def wrong_connection(
        self, circuit: dict, wire_index: Optional[int] = None,
    ) -> tuple[dict, dict]:
        """
        Rewire a connection to a wrong hole.

        Args:
            circuit: Valid circuit config dict.
            wire_index: Index of wire to modify, or None for random.

        Returns:
            (mutated_circuit, mutation_record)
        """
        mutated = copy.deepcopy(circuit)
        wires = mutated['wires']

        if not wires:
            raise ValueError("Circuit has no wires to modify")

        if wire_index is not None:
            idx = wire_index
        else:
            idx = self.rng.randrange(len(wires))

        wire = wires[idx]
        # Pick which end to modify ('from' or 'to')
        end_key = self.rng.choice(['from', 'to'])
        original = copy.deepcopy(wire[end_key])

        # Shift the column by a random amount
        shift = self.rng.choice([s for s in range(-5, 6) if s != 0])
        new_col = wire[end_key][1] + shift
        new_col = max(1, min(self.n_cols, new_col))
        wire[end_key][1] = new_col

        record = {
            "type": "wrong_connection",
            "wire_index": idx,
            "end_modified": end_key,
            "original": original,
            "mutated": copy.deepcopy(wire[end_key]),
        }
        return mutated, record

    def swap_polarity(self, circuit: dict) -> tuple[dict, dict]:
        """
        Swap +/- rail connections (common student mistake).

        Finds wires connected to power rails and swaps + for - or vice versa.

        Returns:
            (mutated_circuit, mutation_record)
        """
        mutated = copy.deepcopy(circuit)
        wires = mutated['wires']

        swap_map = {"p1+": "p1-", "p1-": "p1+", "p2+": "p2-", "p2-": "p2+"}
        swapped = []

        rail_wire_indices = []
        for i, wire in enumerate(wires):
            for end_key in ('from', 'to'):
                if wire[end_key][0] in swap_map:
                    rail_wire_indices.append((i, end_key))

        if not rail_wire_indices:
            raise ValueError("Circuit has no rail connections to swap")

        # Pick one rail connection to swap
        idx, end_key = self.rng.choice(rail_wire_indices)
        wire = wires[idx]
        original = copy.deepcopy(wire[end_key])
        wire[end_key][0] = swap_map[wire[end_key][0]]
        swapped.append({
            "wire_index": idx,
            "end_modified": end_key,
            "original": original,
            "mutated": copy.deepcopy(wire[end_key]),
        })

        record = {
            "type": "swap_polarity",
            "swaps": swapped,
        }
        return mutated, record

    def extra_component(self, circuit: dict) -> tuple[dict, dict]:
        """
        Add a spurious component to the circuit.

        Returns:
            (mutated_circuit, mutation_record)
        """
        mutated = copy.deepcopy(circuit)

        # Generate a random position that doesn't overlap existing components
        existing_cols = set()
        for comp in mutated['components']:
            for pin_val in comp['pins'].values():
                existing_cols.add(pin_val[1])

        # Find a free column range for the extra component
        col = self.rng.randint(1, self.n_cols - 5)
        row = self.rng.choice(_TERMINAL_ROWS[:5])  # top half

        comp_type = self.rng.choice(['resistor', 'led'])
        if comp_type == 'resistor':
            extra = {
                "id": f"EXTRA_R{self.rng.randint(10, 99)}",
                "type": "resistor",
                "value": "1k",
                "pins": {"leg1": [row, col], "leg2": [row, col + 5]},
            }
        else:
            extra = {
                "id": f"EXTRA_LED{self.rng.randint(10, 99)}",
                "type": "led",
                "color": self.rng.choice(["red", "green", "yellow", "blue"]),
                "pins": {"anode": [row, col], "cathode": [chr(ord(row) + 1), col]},
            }

        mutated['components'].append(extra)

        record = {
            "type": "extra_component",
            "added": extra,
        }
        return mutated, record

    def compound_mutation(
        self, circuit: dict, n_mutations: int = 2,
    ) -> tuple[dict, dict]:
        """
        Apply multiple mutations for harder examples.

        Args:
            circuit: Valid circuit config dict.
            n_mutations: Number of mutations to apply.

        Returns:
            (mutated_circuit, mutation_record)
        """
        single_mutations = [
            self.remove_component,
            self.wrong_position,
            self.wrong_connection,
            self.swap_polarity,
            self.extra_component,
        ]

        mutated = copy.deepcopy(circuit)
        records = []

        for _ in range(n_mutations):
            fn = self.rng.choice(single_mutations)
            try:
                mutated, record = fn(mutated)
                records.append(record)
            except (ValueError, IndexError, StopIteration):
                # Skip if mutation can't be applied to current state
                continue

        record = {
            "type": "compound_mutation",
            "n_applied": len(records),
            "mutations": records,
        }
        return mutated, record
