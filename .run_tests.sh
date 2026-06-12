#!/usr/bin/env bash

# ----------------------------------------------------------------------
# -----------------------------  Notes ---------------------------------
#
# I am aware that this script could be replaced by using the "tox" tool
# for example, and that this script assumes that we are on a Unix-like system, etc.
# But:
# - that would be yet another tool to learn, configure and maintain
# - that would be less flexible and transparent: this does exactly what I want it to do
# - it would be hard to find something more stable than a bash script

# ----------------------------------------------------------------------
# --------------------  How to use this script -------------------------
#
# 1) Run tests on all default Python versions (3.11 3.12 3.13 3.14):
#    ./.run_tests.sh
# 2) Run tests on specific versions only:
#    PY_VERSIONS="3.11 3.12" ./.run_tests.sh
# 3) Pass extra pytest options (they are forwarded to pytest):
#    ./.run_tests.sh -q -k "smoke"
# Exit code:
# - 0 if all tested combinations pass
# - 1 if at least one combination fails
# ----------------------------------------------------------------------


# "-u" makes Bash fail on undefined variables.
# "-o pipefail" makes a pipeline fail if any command in it fails.
set -uo pipefail

# Default Python versions to test.
# Override with PY_VERSIONS env var (see usage above).
DEFAULT_PYTHONS=(3.11 3.12 3.13 3.14)

# Dependency resolution strategies to test against.
# "lowest" and "lowest-direct" test against minimum compatible versions;
# "highest" tests against the latest available versions.
# RESOLUTIONS=(highest lowest lowest-direct)
RESOLUTIONS=(highest lowest lowest-direct)

# If PY_VERSIONS is set, split it into an array; otherwise use the defaults.
# shellcheck disable=SC2206
if [[ -n "${PY_VERSIONS:-}" ]]; then
    PYTHONS=(${PY_VERSIONS})
else
    PYTHONS=("${DEFAULT_PYTHONS[@]}")
fi

# Counters used for the final summary.
total=0
passed=0
failed=0

# Capture all output in a logfile; we only print a one-liner per combination.
# This logfile will be deleted if all tests pass, but if something fails, it will be left behind for debugging.
# >>"$log_file" redirects stdout and 2>&1 redirects stderr to the same file (in append mode).
log_file="run_tests-log.txt"
# Reset logfile for this script run so summaries are not polluted by old runs.
: > "$log_file"


for py in "${PYTHONS[@]}"; do
    # Create (or reuse) a dedicated virtual environment for this Python version.
    venv_dir=".venv/test-p${py}"

    for resolution in "${RESOLUTIONS[@]}"; do

        uv venv "$venv_dir" --python "$py" >> "$log_file" 2>&1
        # Activate the venv so the plain `pytest` command uses it.
        source "${venv_dir}/bin/activate"

        # Each (Python version × resolution) combination counts as one test run.
        ((total += 1))


        # Install/sync the dev dependencies using the chosen resolution strategy.
        #
        # Let's decompose the next line of code to understand it better:
        # We use:
        # --active because otherwise, uv uses the default uv venv instead of the one we just created and activated.
        # --group dev because we want to install the dev dependencies (which include pytest)
        # 1) uv sync --active --group dev --resolution "$resolution": installs the correct env
        # 2) >>"$log_file": redirects stdout (anything the command would normally print) to a logfile.
        # 3) 2>&1: redirects stderr (any error messages) to the same place as stdout.
        if ! uv sync --active --group dev --resolution "$resolution" >> "$log_file" 2>&1; then
            printf 'py%s | %-14s: FAIL (uv sync failed)\n' "$py" "$resolution"
            ((failed += 1))
            # Skip to the next resolution; no point running tests without a valid env.
            continue
        fi

        # Run the tests. "$@" forwards any extra args you passed to this script.
        # We use --active because otherwise, uv uses the default uv venv instead of the one we just created and activated.
        if uv run --active pytest "$@" >> "$log_file" 2>&1; then
            rc=0
        else
            rc=$?
        fi

        # Extract pytest's own final summary line, e.g.: === 42 passed in 3.21s ===
        # "tail -n 1" keeps only the last match, in case there are multiple summary lines.
        pytest_summary=$(grep -E "=+ .* in [0-9.]+s =+" "$log_file" | tail -n 1 || true)
        # -z checks if the variable is empty
        if [[ -z "$pytest_summary" ]]; then
            pytest_summary="no pytest summary captured"
        fi

        if [[ $rc -eq 0 ]]; then
            ((passed += 1))
            printf 'py%s | %-14s: PASS | %s\n' "$py" "$resolution" "$pytest_summary"
        else
            ((failed += 1))

            # Try to pull out the most relevant failure line for the one-liner.
            short_reason=$(grep -E "FAILED|ERROR" "$log_file" | tail -n 1 || true)

            if [[ -z "$short_reason" ]]; then
                short_reason="$pytest_summary"
            fi

            printf 'py%s | %-14s: FAIL(rc=%s) | %s\n' "$py" "$resolution" "$rc" "$short_reason"
        fi

        # Deactivate the venv before moving to the next Python version.
        deactivate

    done

    # Delete the test venv for cleanliness
    rm -rf "$venv_dir"
done

# Final summary and exit code.
if [[ $failed -eq 0 ]]; then
    # Exit code 0 = success.
    echo "ALL TESTS PASSED ($passed/$total)"

    # We can delete logfile for this run, since tests passed (for cleanliness)
    rm -f "$log_file"

    exit 0
else
    # Exit code 1 = at least one combination failed.
    echo "TESTS FAILED ($failed failed, $passed passed, $total total)"
    exit 1
fi
