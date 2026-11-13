"""The six leakage audits must pass on the shipped frozen splits."""
import audit_splits


def test_all_six_audits_pass():
    results = audit_splits.run_audits()
    assert len(results) == 6
    failing = [name for name, violations, _ in results if violations]
    assert failing == []


def test_jzo_rows_present_on_time_test():
    df = audit_splits.load()
    jzo = df[df["fw_key"] == "JZO"]
    assert len(jzo) == 32
    assert (jzo["split_time_2018"] == "test").all()


def test_time_boundaries():
    df = audit_splits.load()
    for cutoff in (2018, 2020):
        col = f"split_time_{cutoff}"
        assert df.loc[df[col] == "train", "year"].max() <= cutoff
        assert df.loc[df[col] == "test", "year"].min() > cutoff
