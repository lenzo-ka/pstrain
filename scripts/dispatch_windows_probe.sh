#!/usr/bin/env bash
set -euo pipefail

branch=${1:-$(git branch --show-current)}
if [[ -z "$branch" ]]; then
    echo "usage: $0 [branch]" >&2
    exit 2
fi

head_sha=$(git rev-parse "$branch^{commit}")
short_sha=$(git rev-parse --short "$head_sha")
probe_branch="probe/${branch//\//-}-$short_sha"
probe_ref="refs/heads/$probe_branch"
probe_sha=$(printf '%s\n' "Windows probe for $branch at $short_sha" |
    git commit-tree "$head_sha^{tree}" -p "$head_sha")

# shellcheck disable=SC2329 # invoked by the EXIT trap
cleanup() {
    git push origin --delete "$probe_branch" >/dev/null 2>&1 || true
    git update-ref -d "$probe_ref" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git update-ref "$probe_ref" "$probe_sha"
git push origin "$probe_ref:$probe_ref"
gh workflow run windows-scoping.yml --ref "$probe_branch"

for _ in {1..30}; do
    run=$(gh run list --workflow windows-scoping.yml --branch "$probe_branch" \
        --event workflow_dispatch --limit 1 \
        --json headSha,url --jq '.[0] | select(.headSha == "'"$probe_sha"'") | .url')
    if [[ -n "$run" ]]; then
        echo "Windows probe dispatched for $probe_sha"
        echo "$run"
        exit 0
    fi
    sleep 2
done

echo "Workflow dispatch was accepted, but its run was not confirmed within 60 seconds" >&2
exit 1
