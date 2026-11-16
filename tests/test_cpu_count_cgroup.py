import os
import sys

import pytest

from loky.backend import context

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Cgroups are specific to Linux"
)

SCOPE = "/user.slice/user-1000.slice/user@1000.service/app.slice/run-u7.scope"


def v1_quota(cgroup_dir, quota):
    return {
        f"/sys/fs/cgroup/cpu{cgroup_dir}/cpu.cfs_quota_us": f"{quota}\n",
        f"/sys/fs/cgroup/cpu{cgroup_dir}/cpu.cfs_period_us": "100000\n",
    }


@pytest.mark.parametrize(
    "proc_cgroup,files,expected",
    [
        pytest.param(
            f"0::{SCOPE}\n",
            {f"/sys/fs/cgroup{SCOPE}/cpu.max": "200000 100000\n"},
            2,
            id="v2-systemd-scope",
        ),
        pytest.param(
            f"0::{SCOPE}\n",
            {
                f"/sys/fs/cgroup{SCOPE}/cpu.max": "max 100000\n",
                "/sys/fs/cgroup/user.slice/cpu.max": "300000 100000\n",
            },
            3,
            id="v2-quota-on-parent",
        ),
        pytest.param(
            f"0::{SCOPE}\n",
            {
                f"/sys/fs/cgroup{SCOPE}/cpu.max": "400000 100000\n",
                "/sys/fs/cgroup/user.slice/cpu.max": "150000 100000\n",
            },
            2,
            id="v2-smallest-quota-rounded-up",
        ),
        pytest.param(
            "0::/\n",
            {"/sys/fs/cgroup/cpu.max": "150000 100000\n"},
            2,
            id="v2-container",
        ),
        pytest.param(
            "0::/system.slice/docker-1a2b.scope\n",
            {
                "/sys/fs/cgroup/system.slice/docker-1a2b.scope/cpu.max": (
                    "100000 100000\n"
                ),
            },
            1,
            id="v2-container-with-host-cgroupns",
        ),
        pytest.param(
            "0::/../other\n",
            {
                "/sys/fs/cgroup/cpu.max": "max 100000\n",
                # Not an ancestor: the cgroup of the process is not visible.
                "/sys/fs/other/cpu.max": "100000 100000\n",
            },
            8,
            id="v2-cgroup-outside-of-cgroupns",
        ),
        pytest.param(
            "5:cpuset:/other\n"
            "4:cpu,cpuacct:/system.slice/foo.service\n"
            "1:name=systemd:/system.slice/foo.service\n",
            {
                **v1_quota("", -1),
                **v1_quota("/system.slice/foo.service", 200000),
                # Not an ancestor: the cgroup of another controller.
                **v1_quota("/other", 100000),
            },
            2,
            id="v1-systemd-service",
        ),
        pytest.param(
            # The root of the mount is the cgroup of the container.
            "4:cpu,cpuacct:/docker/1a2b\n",
            {
                **v1_quota("", 150000),
                # Not an ancestor: a child cgroup of the container.
                **v1_quota("/docker", 100000),
            },
            2,
            id="v1-container",
        ),
        pytest.param(
            f"4:cpu,cpuacct:{SCOPE}\n0::{SCOPE}\n",
            v1_quota(SCOPE, 200000),
            2,
            id="hybrid",
        ),
        pytest.param(
            f"0::{SCOPE}\n",
            {f"/sys/fs/cgroup{SCOPE}/cpu.max": "max 100000\n"},
            8,
            id="no-quota",
        ),
        pytest.param(
            "unexpected\n",
            {"/sys/fs/cgroup/cpu.max": "150000 100000\n"},
            2,
            id="unexpected-proc-cgroup",
        ),
        pytest.param(
            None,
            {"/sys/fs/cgroup/cpu.max": "150000 100000\n"},
            2,
            id="v2-without-proc",
        ),
        pytest.param(None, v1_quota("", 150000), 2, id="v1-without-proc"),
        pytest.param(None, {}, 8, id="no-cgroup"),
    ],
)
def test_cpu_count_cgroup(tmp_path, monkeypatch, proc_cgroup, files, expected):
    if proc_cgroup is not None:
        files = {**files, "/proc/self/cgroup": proc_cgroup}
    for path, content in files.items():
        file = tmp_path / path.lstrip("/")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)

    # Use the files above instead of the ones of this machine.
    def local(path):
        return tmp_path / path.lstrip("/")

    real_open, real_isdir = open, os.path.isdir
    monkeypatch.setattr(
        context,
        "open",
        lambda path, *args, **kwargs: real_open(local(path), *args, **kwargs),
        raising=False,
    )
    monkeypatch.setattr(os.path, "isdir", lambda path: real_isdir(local(path)))
    assert context._cpu_count_cgroup(8) == expected
