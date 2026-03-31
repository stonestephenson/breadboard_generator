"""
Circuit config loader and render orchestrator.

Reads a circuit JSON config and renders all components and wires onto a
blank board. No drawing logic here — delegates to components.py and wires.py.
"""

import json
import os
from PIL import Image
from generator.grid import BreadboardGrid, load_spec
from generator.board import draw_board_base
from generator.holes import draw_holes
from generator.wires import draw_wire
from generator.components import Resistor, LED


def load_circuit(circuit_path: str) -> dict:
    """Load a circuit config from JSON file."""
    with open(circuit_path) as f:
        return json.load(f)


def _pin(pin_json: list) -> tuple[str, int]:
    """Convert a JSON [row, col] array to a (row, col) tuple."""
    return (pin_json[0], pin_json[1])


def _build_component(comp: dict) -> Resistor | LED:
    """Instantiate a component object from a circuit config entry."""
    ctype = comp['type']

    if ctype == 'resistor':
        pins = comp['pins']
        bands = [tuple(b) for b in comp['bands']] if 'bands' in comp else None
        return Resistor(leg1=_pin(pins['leg1']), leg2=_pin(pins['leg2']), bands=bands)

    elif ctype == 'led':
        pins = comp['pins']
        color = comp.get('color', 'red')
        return LED(anode=_pin(pins['anode']), cathode=_pin(pins['cathode']), color=color)

    else:
        raise ValueError(f"Unknown component type: {ctype!r}")


def render_circuit(
    circuit_config: dict,
    spec: dict,
    ppmm_override: float | None = None,
) -> Image.Image:
    """
    Render a circuit onto a blank board.

    Args:
        circuit_config: Parsed circuit JSON dict.
        spec: Parsed board_spec.json dict.
        ppmm_override: Override pixels_per_mm (for supersampling).

    Returns:
        PIL Image with the circuit rendered.
    """
    grid = BreadboardGrid(spec, ppmm_override=ppmm_override)

    img = draw_board_base(grid)
    img = draw_holes(img, grid)

    # Draw components
    for comp_def in circuit_config.get('components', []):
        component = _build_component(comp_def)
        img = component.draw(img, grid)

    # Draw wires
    for wire_def in circuit_config.get('wires', []):
        img = draw_wire(
            img, grid,
            start=_pin(wire_def['from']),
            end=_pin(wire_def['to']),
            color=wire_def.get('color', 'red'),
            routing=wire_def.get('routing', 'straight'),
        )

    return img


def render_circuit_to_file(
    circuit_path: str,
    spec_path: str,
    output_path: str,
) -> Image.Image:
    """
    Load a circuit config and board spec, render, and save to file.

    Renders at supersample resolution, then downscales for antialiasing.

    Args:
        circuit_path: Path to circuit JSON config.
        spec_path: Path to board_spec.json.
        output_path: Path for output PNG.

    Returns:
        Final PIL Image at target resolution.
    """
    spec = load_spec(spec_path)
    circuit_config = load_circuit(circuit_path)
    ss = spec['rendering']['supersample_factor']
    base_ppmm = spec['board']['pixels_per_mm']

    img = render_circuit(circuit_config, spec, ppmm_override=base_ppmm * ss)

    # Downscale to target resolution
    if ss > 1:
        target_w = round(spec['board']['real_width_mm'] * base_ppmm)
        target_h = round(spec['board']['real_height_mm'] * base_ppmm)
        img = img.resize((target_w, target_h), Image.LANCZOS)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    return img
