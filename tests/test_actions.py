import pytest

from autofill_agent.actions import Click, Fill, Key, Select, Type, check_policy, parse_actions


def test_parses_each_action_type():
    actions = parse_actions([
        {"type": "fill", "ref": "e1", "value": "Ada"},
        {"type": "select", "ref": "e2", "label": "Germany"},
        {"type": "check", "ref": "e3"},
        {"type": "click", "ref": "e4", "mode": "dom"},
        {"type": "type", "text": "hello"},
        {"type": "key", "key": "Tab"},
        {"type": "scroll", "dy": 300},
        {"type": "wait", "ms": 100},
    ])
    assert [a.type for a in actions] == ["fill", "select", "check", "click", "type", "key", "scroll", "wait"]
    assert actions[3].mode == "dom"
    assert actions[2].checked is True


def test_rejects_unknown_action_and_bad_fields():
    with pytest.raises(Exception):
        parse_actions([{"type": "drop_table", "ref": "e1"}])
    with pytest.raises(Exception):
        parse_actions([{"type": "wait", "ms": 999999}])
    with pytest.raises(Exception):
        parse_actions([{"type": "select", "ref": "e2"}])  # needs value or label


def test_batch_size_is_limited():
    with pytest.raises(ValueError):
        parse_actions([{"type": "wait", "ms": 1}] * 21)


def test_submit_click_is_blocked():
    reason = check_policy(Click(type="click", ref="e9"), {"label": "Submit application", "input_type": "submit"})
    assert reason and "approval" in reason


def test_normal_click_is_allowed():
    assert check_policy(Click(type="click", ref="e9"), {"label": "Add another role", "input_type": "button"}) is None


def test_password_fill_is_blocked():
    assert check_policy(Fill(type="fill", ref="e1", value="x"), {"input_type": "password"}) is not None


def test_file_fill_is_blocked():
    assert check_policy(Fill(type="fill", ref="e1", value="x"), {"input_type": "file"}) is not None


def test_enter_key_and_newline_are_blocked():
    assert check_policy(Key(type="key", key="Enter")) is not None
    assert check_policy(Type(type="type", text="line\nnext")) is not None
    assert check_policy(Key(type="key", key="Tab")) is None


def test_select_requires_value_or_label():
    with pytest.raises(Exception):
        Select(type="select", ref="e1")
