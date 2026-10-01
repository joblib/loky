import os

import pytest

from loky.backend import context


@pytest.fixture
def cgroup_files(tmp_path, monkeypatch):
    """Expose real fixture files through the Linux procfs/cgroup paths."""
    real_open = open
    real_exists = os.path.exists

    def local_path(path):
        path = os.fspath(path)
        if path.startswith(("/proc/self/", "/sys/")):
            if os.name == "nt" and any(c in path for c in "\t\n\r\\"):
                pytest.skip("Fixture uses POSIX-only filename characters")
            return tmp_path / path.lstrip("/")
        return path

    def fixture_open(path, *args, **kwargs):
        return real_open(local_path(path), *args, **kwargs)

    def fixture_exists(path):
        return real_exists(local_path(path))

    monkeypatch.setattr(context, "open", fixture_open, raising=False)
    monkeypatch.setattr(os.path, "exists", fixture_exists)

    def write(path, content):
        target = local_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content = os.fsencode(content)
        # Procfs emits literal LF bytes, even when tests run on Windows.
        target.write_bytes(content)

    return write


def mountinfo(root="/", mountpoint="/sys/fs/cgroup"):
    return (
        f"29 23 0:26 {root} {mountpoint} rw,nosuid,nodev,noexec "
        "shared:4 - cgroup2 cgroup rw\n"
    )


@pytest.mark.parametrize(
    "membership,leaf,parent,expected",
    [
        ("/user.slice/app.scope", "200000 100000", None, 2),
        ("/user.slice/app.scope", "150000 100000", None, 2),
        ("/user.slice/app.scope", "max 100000", "200000 100000", 2),
        ("/user.slice/app.scope", "300000 100000", "150000 100000", 2),
        ("/user.slice/app.scope", "100000 100000", "200000 100000", 1),
        ("/user.slice/app.scope", "max 100000", "max 100000", 8),
        ("/", "200000 100000", None, 2),
    ],
)
def test_cgroup_v2_process_quota(
    cgroup_files, membership, leaf, parent, expected
):
    cgroup_files("/proc/self/cgroup", f"1:net_cls:/\n0::{membership}\n")
    cgroup_files("/proc/self/mountinfo", mountinfo())
    cgroup_files(f"/sys/fs/cgroup{membership}/cpu.max", leaf)
    if parent is not None:
        cgroup_files("/sys/fs/cgroup/user.slice/cpu.max", parent)
    assert context._cpu_count_cgroup(8) == expected


@pytest.mark.parametrize(
    "root,mountpoint,membership",
    [
        ("/user.slice", "/sys/delegated", "/user.slice/app.scope"),
        ("/", "/sys/unified", "/app.scope"),
        ("/tenant", "/sys/delegated", "/tenant"),
    ],
)
def test_cgroup_v2_mount_root(cgroup_files, root, mountpoint, membership):
    cgroup_files("/proc/self/cgroup", f"0::{membership}\n")
    cgroup_files("/proc/self/mountinfo", mountinfo(root, mountpoint))
    relative = membership[len(root) :].lstrip("/")
    cgroup_files(f"{mountpoint}/{relative}/cpu.max", "200000 100000")
    assert context._cpu_count_cgroup(8) == 2


@pytest.mark.parametrize("metadata", [None, "0::/\n", "1:cpu:/app\n"])
def test_cgroup_root_fallback(cgroup_files, metadata):
    if metadata is not None:
        cgroup_files("/proc/self/cgroup", metadata)
    cgroup_files("/sys/fs/cgroup/cpu.max", "150000 100000")
    assert context._cpu_count_cgroup(8) == 2


@pytest.mark.parametrize("quota,expected", [("150000", 2), ("-1", 8)])
def test_cgroup_v1_unchanged(cgroup_files, quota, expected):
    cgroup_files("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", quota)
    cgroup_files("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "100000")
    assert context._cpu_count_cgroup(8) == expected


def test_cgroup_no_limit(cgroup_files):
    assert context._cpu_count_cgroup(8) == 8


@pytest.mark.parametrize(
    "affinity,env,expected", [(8, "8", 2), (1, "8", 1), (8, "1", 1)]
)
def test_cgroup_public_cpu_count(
    cgroup_files, monkeypatch, affinity, env, expected
):
    cgroup_files("/proc/self/cgroup", "0::/app\n")
    cgroup_files("/proc/self/mountinfo", mountinfo())
    cgroup_files("/sys/fs/cgroup/app/cpu.max", "200000 100000")
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        context, "_cpu_count_affinity_set", lambda: set(range(affinity))
    )
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", env)
    assert context.cpu_count() == expected


@pytest.mark.parametrize("character", [" ", "\t", "\n", "\\"])
def test_cgroup_v2_escaped_mount_paths(cgroup_files, character):
    # The membership path is literal; mountinfo escapes these characters.
    root = "/tenant" + character + "name"
    mountpoint = "/sys/cgroup" + character + "mount"
    escape = f"\\{ord(character):03o}"
    # Exercise newline escaping only in mountinfo; membership is line-oriented.
    membership = "/app" if character == "\n" else root + "/app"
    mount_root = "/" if character == "\n" else root
    cgroup_files("/proc/self/cgroup", f"0::{membership}\n")
    cgroup_files(
        "/proc/self/mountinfo",
        mountinfo(
            mount_root.replace(character, escape),
            mountpoint.replace(character, escape),
        ),
    )
    cgroup_files(f"{mountpoint}/app/cpu.max", "100000 100000")
    assert context._cpu_count_cgroup(8) == 1


@pytest.mark.parametrize("membership", ["/tenant2/app", "/../app", "relative"])
def test_cgroup_v2_unreachable_membership(cgroup_files, membership):
    cgroup_files("/proc/self/cgroup", f"0::{membership}\n")
    cgroup_files(
        "/proc/self/mountinfo", mountinfo("/tenant", "/sys/delegated")
    )
    cgroup_files("/sys/delegated/cpu.max", "100000 100000")
    assert context._cpu_count_cgroup(8) == 8


def test_cgroup_v2_does_not_walk_above_mount(cgroup_files):
    cgroup_files("/proc/self/cgroup", "0::/tenant/app\n")
    cgroup_files(
        "/proc/self/mountinfo", mountinfo("/tenant", "/sys/delegated")
    )
    cgroup_files("/sys/delegated/app/cpu.max", "300000 100000")
    cgroup_files("/sys/cpu.max", "100000 100000")
    assert context._cpu_count_cgroup(8) == 3


@pytest.mark.parametrize("v2_contents", ["", "max", "max 100000"])
def test_cgroup_v2_malformed_falls_back_to_v1(cgroup_files, v2_contents):
    cgroup_files("/sys/fs/cgroup/cpu.max", v2_contents)
    cgroup_files("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "100000")
    cgroup_files("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "100000")
    assert context._cpu_count_cgroup(8) == (
        8 if v2_contents == "max 100000" else 1
    )


def test_cgroup_v2_ignores_other_mounts(cgroup_files):
    cgroup_files("/proc/self/cgroup", "0::/app\n")
    cgroup_files(
        "/proc/self/mountinfo",
        "malformed\n"
        + mountinfo(mountpoint="/sys/wrong").replace(
            " - cgroup2 ", " - tmpfs "
        )
        + mountinfo(),
    )
    cgroup_files("/sys/wrong/app/cpu.max", "100000 100000")
    cgroup_files("/sys/fs/cgroup/app/cpu.max", "200000 100000")
    assert context._cpu_count_cgroup(8) == 2


@pytest.mark.parametrize(
    "blocked", ["/proc/self/cgroup", "/proc/self/mountinfo"]
)
def test_cgroup_v2_metadata_unavailable(cgroup_files, monkeypatch, blocked):
    cgroup_files("/proc/self/cgroup", "0::/app\n")
    cgroup_files("/proc/self/mountinfo", mountinfo())
    cgroup_files("/sys/fs/cgroup/cpu.max", "200000 100000")
    fixture_open = context.open

    def denied_metadata(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError(path)
        return fixture_open(path, *args, **kwargs)

    monkeypatch.setattr(context, "open", denied_metadata)
    assert context._cpu_count_cgroup(8) == 2


def test_cgroup_v2_missing_leaf_keeps_parent_limit(cgroup_files):
    cgroup_files("/proc/self/cgroup", "0::/user.slice/app.scope\n")
    cgroup_files("/proc/self/mountinfo", mountinfo())
    cgroup_files("/sys/fs/cgroup/user.slice/cpu.max", "150000 100000")
    assert context._cpu_count_cgroup(8) == 2


def test_cgroup_v2_multiple_mount_views(cgroup_files):
    cgroup_files("/proc/self/cgroup", "0::/tenant/app\n")
    cgroup_files(
        "/proc/self/mountinfo",
        mountinfo("/tenant/app", "/sys/delegated").replace("29 23 ", "30 23 ")
        + mountinfo(),
    )
    cgroup_files("/sys/delegated/cpu.max", "max 100000")
    cgroup_files("/sys/fs/cgroup/tenant/cpu.max", "200000 100000")
    assert context._cpu_count_cgroup(8) == 2


def test_cgroup_v2_stacked_unrelated_mount(cgroup_files):
    cgroup_files("/proc/self/cgroup", "0::/tenantA/app\n")
    upper = mountinfo("/tenantB").replace("29 23 ", "30 29 ")
    cgroup_files("/proc/self/mountinfo", mountinfo() + upper)
    # These files now belong to tenantB, not the hidden lower mount.
    cgroup_files("/sys/fs/cgroup/cpu.max", "max 100000")
    cgroup_files("/sys/fs/cgroup/tenantA/app/cpu.max", "100000 100000")
    assert context._cpu_count_cgroup(8) == 8


def test_cgroup_v2_stacked_matching_mount(cgroup_files):
    cgroup_files("/proc/self/cgroup", "0::/tenantA/app\n")
    upper = mountinfo("/tenantA").replace("29 23 ", "30 29 ")
    cgroup_files("/proc/self/mountinfo", mountinfo() + upper)
    cgroup_files("/sys/fs/cgroup/app/cpu.max", "200000 100000")
    # The path suggested by the hidden lower mount is unrelated.
    cgroup_files("/sys/fs/cgroup/tenantA/app/cpu.max", "100000 100000")
    assert context._cpu_count_cgroup(8) == 2


@pytest.mark.parametrize("covered", ["/tenant/app", "/tenant/app/cpu.max"])
def test_cgroup_v2_non_cgroup_overmount(cgroup_files, covered):
    cgroup_files("/proc/self/cgroup", "0::/tenant/app\n")
    overmount = f"30 29 0:27 / /sys/fs/cgroup{covered} rw - tmpfs tmpfs rw\n"
    cgroup_files("/proc/self/mountinfo", mountinfo() + overmount)
    cgroup_files("/sys/fs/cgroup/tenant/app/cpu.max", "100000 100000")
    cgroup_files("/sys/fs/cgroup/tenant/cpu.max", "300000 100000")
    assert context._cpu_count_cgroup(8) == 3


def test_cgroup_v2_nested_mount_hidden_by_ancestor(cgroup_files):
    cgroup_files("/proc/self/cgroup", "0::/tenant/app\n")
    nested = mountinfo("/tenant", "/sys/fs/cgroup/alias").replace(
        "29 23 ", "30 29 "
    )
    upper = "31 29 0:27 / /sys/fs/cgroup rw - tmpfs tmpfs rw\n"
    cgroup_files("/proc/self/mountinfo", mountinfo() + nested + upper)
    cgroup_files("/sys/fs/cgroup/alias/app/cpu.max", "100000 100000")
    assert context._cpu_count_cgroup(8) == 8


@pytest.mark.parametrize("character", ["\u00a0", "\r"])
def test_cgroup_v2_unescaped_whitespace(cgroup_files, character):
    root = f"/tenant{character}name"
    mountpoint = f"/sys/cg{character}mount"
    cgroup_files("/proc/self/cgroup", f"0::{root}/app\n")
    cgroup_files("/proc/self/mountinfo", mountinfo(root, mountpoint))
    cgroup_files(f"{mountpoint}/app/cpu.max", "200000 100000")
    assert context._cpu_count_cgroup(8) == 2


@pytest.mark.skipif(
    os.name == "nt", reason="Arbitrary byte paths require POSIX"
)
def test_cgroup_v2_non_utf8_paths(cgroup_files):
    root = os.fsdecode(b"/tenant\xff")
    mountpoint = "/sys/delegated"
    cgroup_files("/proc/self/cgroup", os.fsencode(f"0::{root}/app\n"))
    cgroup_files(
        "/proc/self/mountinfo", os.fsencode(mountinfo(root, mountpoint))
    )
    cgroup_files(f"{mountpoint}/app/cpu.max", "200000 100000")
    assert context._cpu_count_cgroup(8) == 2


@pytest.mark.parametrize(
    "membership", ["0::/foo\nbar\n", "0::/foo\n\n", "0::/foo\n0::/bar\n"]
)
def test_cgroup_v2_ambiguous_membership_fallback(cgroup_files, membership):
    cgroup_files("/proc/self/cgroup", membership)
    cgroup_files("/proc/self/mountinfo", mountinfo())
    cgroup_files("/sys/fs/cgroup/foo/cpu.max", "100000 100000")
    cgroup_files("/sys/fs/cgroup/cpu.max", "max 100000")
    assert context._cpu_count_cgroup(8) == 8
