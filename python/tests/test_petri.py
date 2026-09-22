import pytest
import torch

from zrth import X
from zrth.eval import execute_init, execute_update
from zrth.petri import load_sdcpn

QUEUE = """
version: 1
title: bounded queue
places:
  - {id: p_queue, name: Queue, colorId: null, dynamicsEnabled: false}
  - {id: p_free, name: Free, colorId: null, dynamicsEnabled: false}
transitions:
  - id: t_arrive
    name: arrive
    inputArcs: [{placeId: p_free, weight: 1, type: standard}]
    outputArcs: [{placeId: p_queue, weight: 1}]
    lambdaType: predicate
    lambdaCode: export default Lambda(() => true);
    metadata: {control: controllable}
  - id: t_serve
    name: serve
    inputArcs: [{placeId: p_queue, weight: 1, type: standard}]
    outputArcs: [{placeId: p_free, weight: 1}]
    lambdaType: predicate
    lambdaCode: export default Lambda(() => true);
scenarios:
  - id: s_empty
    name: empty
    initialState: {type: per_place, content: {p_queue: '0', p_free: '3'}}
"""


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "queue.yaml"
    path.write_text(QUEUE)
    return load_sdcpn(path)


def _reset(net):
    state = {}
    execute_init(state, net.module.atoms)
    return {name: state[X(v)] for name, v in net.places.items()}


def _fire(net, marking, index):
    state = {net.places[name]: value for name, value in marking.items()}
    state[net.fire] = torch.tensor([[index]])
    execute_update(state, net.module.atoms)
    return {name: state[X(v)] for name, v in net.places.items()}


def _counts(marking):
    return {name: int(t.item()) for name, t in marking.items()}


def test_structure(net):
    assert net.module.open()
    assert set(net.places) == {"Queue", "Free"}
    assert net.transitions == ["arrive", "serve"]
    assert net.control == {"arrive": "controllable", "serve": None}
    assert net.marking == {"Queue": 0, "Free": 3}
    assert net.pre == [{"Free": 1}, {"Queue": 1}]
    assert net.post == [{"Queue": 1}, {"Free": 1}]


def test_firing_moves_tokens(net):
    m = _reset(net)
    assert _counts(m) == {"Queue": 0, "Free": 3}
    m = _fire(net, m, 0)
    assert _counts(m) == {"Queue": 1, "Free": 2}
    m = _fire(net, m, 1)
    assert _counts(m) == {"Queue": 0, "Free": 3}


def test_disabled_or_unknown_choice_stutters(net):
    m = _reset(net)
    assert _counts(_fire(net, m, 1)) == {"Queue": 0, "Free": 3}
    assert _counts(_fire(net, m, 7)) == {"Queue": 0, "Free": 3}


def test_unknown_scenario_and_nontrivial_lambda(tmp_path):
    path = tmp_path / "queue.yaml"
    path.write_text(QUEUE)
    with pytest.raises(KeyError):
        load_sdcpn(path, scenario="missing")
    path.write_text(QUEUE.replace("() => true", "(input) => input.Free.count > 1", 1))
    with pytest.raises(NotImplementedError):
        load_sdcpn(path)
