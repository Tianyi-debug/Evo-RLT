from evo_rlt.cli.audit_online_takeovers import (
    CORRECTIVE,
    PROACTIVE,
    EpisodeAudit,
    _anchor_count,
    episode_split,
    future_k_counts,
)


def _episode(uid: str, category: str, anchors: int) -> EpisodeAudit:
    assisted = category in {"corrective", "proactive"}
    reason = CORRECTIVE if category == "corrective" else PROACTIVE if assisted else 0
    success = category != "autonomous_failure"
    return EpisodeAudit(
        uid=uid,
        dataset="dataset",
        episode_index=int(uid.split("-")[-1]),
        success=success,
        frame_count=100,
        assisted=assisted,
        event_count=int(assisted),
        reason=reason,
        prefix_frames=50 if assisted else 100,
        prefix_anchors=anchors,
        return_to_policy=False,
    )


def test_anchor_count_keeps_action_and_bootstrap_before_boundary():
    assert _anchor_count(frame_count=23, chunk_length=10, stride=2) == 7
    assert _anchor_count(frame_count=10, chunk_length=10, stride=2) == 0


def test_future_k_respects_corrective_proactive_and_failure_rules():
    episodes = [
        _episode("ep-0", "corrective", 8),
        _episode("ep-1", "proactive", 7),
        _episode("ep-2", "autonomous_success", 10),
        _episode("ep-3", "autonomous_failure", 9),
    ]
    counts = future_k_counts(episodes, k=3)
    assert counts["positive"] == 3
    assert counts["negative"] == 5 + 4 + 10
    assert counts["censored"] == 3
    assert counts["autonomous_failure_excluded"] == 9


def test_episode_split_never_splits_an_episode_and_stratifies_events():
    episodes = [
        _episode(f"corrective-{index}", "corrective", 8) for index in range(10)
    ] + [
        _episode(f"success-{index + 10}", "autonomous_success", 10)
        for index in range(10)
    ]
    train, val = episode_split(episodes, val_fraction=0.2, seed=1000)
    assert len(train) == 16
    assert len(val) == 4
    assert len({item.uid for item in train} & {item.uid for item in val}) == 0
    assert sum(item.category == "corrective" for item in val) == 2
