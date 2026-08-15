#!/usr/bin/env bash
set -euo pipefail

branch=${1:-$(git branch --show-current)}
if [[ -z "$branch" ]]; then
    echo "usage: $0 [branch]" >&2
    exit 2
fi

head_sha=$(git rev-parse "$branch^{commit}")
short_sha=$(git rev-parse --short "$head_sha")
probe_sha=$(printf '%s\n' "Windows probe for $branch at $short_sha" |
    git commit-tree "$head_sha^{tree}" -p "$head_sha")
probe_tag="probe-${branch//\//-}-$short_sha-$(date -u +%Y%m%dT%H%M%SZ)-$$"
probe_ref="refs/tags/$probe_tag"

# shellcheck disable=SC2329 # invoked by the EXIT trap
cleanup() {
    git push origin ":$probe_ref" >/dev/null 2>&1 || true
    git update-ref -d "$probe_ref" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git update-ref "$probe_ref" "$probe_sha"
git push origin "$probe_ref:$probe_ref"
gh workflow run windows-scoping.yml --ref "$probe_tag"

for _ in {1..30}; do
    run=$(gh run list --workflow windows-scoping.yml --branch "$probe_tag" \
        --event workflow_dispatch --limit 1 \
        --json databaseId,headSha,url --jq '.[0] | [.databaseId, .headSha, .url] | @tsv')
    if [[ -n "$run" ]]; then
        IFS=$'\t' read -r run_id run_sha run_url <<<"$run"
        if [[ "$run_sha" != "$probe_sha" ]]; then
            gh run cancel "$run_id" >/dev/null 2>&1 || true
            echo "Windows probe SHA mismatch: expected $probe_sha, created run $run_id has $run_sha; run canceled" >&2
            exit 1
        fi
        echo "Windows probe dispatched for $probe_sha"
        echo "$run_url"
        exit 0
    fi
    sleep 2
done

echo "Workflow dispatch was accepted, but its run was not confirmed within 60 seconds" >&2
exit 1
