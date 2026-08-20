# Maintaining the stdlib vendored resource tracker

For easier maintainability, the idea is that Loky resource tracker in
`loky/backend/resource_tracker.py` derives from a minimally modified version of
the stdlib resource tracker for a fixed CPython version in
`loky/backend/stdlib_py314_resource_tracker.py`. Loky-specific features (like
Windows support, logging, refcount functionality for shared folders and files,
etc ...) should go as much as possible in `loky/backend/resource_tracker.py`.

`vendor_stdlib_resource_tracker.sh` downloads the `resource_tracker.py` file
from a fixed CPython version. Loky minimal changes to that file should be
stored as a patch in `tools/stdlib_py314_resource_tracker.patch`. As a patch, it
has more chances to still apply if the CPython code change and we want to
update the vendored stdlib copy.

## Running the vendoring script

Typical use case: update `loky/backend/stdlib_py314_resource_tracker.py` to
mirror the changes in CPython from 3.14.7 to 3.15.x. If the patch still applies
the vendoring script will update
`loky/backend/stdlib_py314_resource_tracker.py`.

- update the CPython version we are fetching `resource_tracker.py` from
- run the vendoring script from the repository root:
```bash
bash tools/vendor_stdlib_resource_tracker.sh
```

## Regenerating the patch

- modify `loky/backend/stdlib_py314_resource_tracker.py` if really needed
  (remember changes should be minimal to avoid diverging from the reference
  CPython code)
- Download the Python reference file
```bash
curl -L \
    https://raw.githubusercontent.com/python/cpython/refs/tags/v3.14.7/Lib/multiprocessing/resource_tracker.py \
    -o loky/backend/stdlib_py314_resource_tracker.py.ref
```
- generate the patch
```bash
diff -u loky/backend/stdlib_py314_resource_tracker.py{.ref,} > tools/stdlib_py314_resource_tracker.patch
```
