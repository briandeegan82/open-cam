"""Tests for the demo registry and argument parsing in ``opencam_gui.ui.cli``."""

from __future__ import annotations

import pytest
from opencam_gui.ui import cli

BUILT_TOPICS = tuple(t.id for t in cli.TOPICS)


def test_every_topic_id_is_unique():
    ids = [t.id for t in cli.TOPICS]
    assert len(ids) == len(set(ids))


def test_no_alias_collides_between_topics():
    seen: dict[str, str] = {}
    for topic in cli.TOPICS:
        for name in topic.command_names:
            assert name not in seen, f"{name!r} claimed by both {seen.get(name)} and {topic.id}"
            seen[name] = topic.id


def test_topics_are_registered_in_teaching_order():
    """The registry order is what ``opencam-gui topics`` prints, so it has to
    match the tutorial sequence."""
    assert [t.id for t in cli.TOPICS] == [
        "geometry",
        "optics",
        "mtf",
        "sensor",
        "exposure",
        "isp",
        "image_generation",
    ]


@pytest.mark.parametrize("topic_id", BUILT_TOPICS)
def test_canonical_name_and_aliases_resolve_to_the_same_topic(topic_id):
    topic = next(t for t in cli.TOPICS if t.id == topic_id)
    for name in topic.command_names:
        assert cli.TOPIC_BY_NAME[name] is topic


def test_underscore_topic_id_is_exposed_with_dashes():
    assert "image-generation" in cli.TOPIC_BY_NAME
    assert cli.TOPIC_BY_NAME["image-generation"].id == "image_generation"


@pytest.mark.parametrize("topic_id", BUILT_TOPICS)
def test_list_scenarios_loads_the_real_scenario_module(topic_id):
    topic = next(t for t in cli.TOPICS if t.id == topic_id)
    scenarios = topic.list_scenarios()
    assert scenarios, f"{topic_id} exposes no scenarios"
    for sc in scenarios:
        assert sc.id and sc.title and sc.teaching_point


@pytest.mark.parametrize("topic_id", BUILT_TOPICS)
def test_list_scenarios_flag_prints_and_exits_without_a_window(capsys, topic_id):
    rc = cli.main(["demo", topic_id.replace("_", "-"), "--list-scenarios"])
    assert rc == 0
    assert capsys.readouterr().out.strip()


def test_topics_command_lists_every_demo(capsys):
    assert cli.main(["topics"]) == 0
    out = capsys.readouterr().out
    for topic in cli.TOPICS:
        assert topic.title in out


def test_unknown_topic_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        cli.main(["demo", "not-a-real-topic"])
