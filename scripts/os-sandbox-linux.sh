#!/bin/sh
# Run the OS-sandbox probes (tests/test_os_sandbox.py) under bubblewrap, in a
# throwaway Debian container, as an ordinary user. srt needs Node 20.10 or
# later, so the base is the official Node 22 image. The repository is mounted
# read-only and copied in, so nothing is written back.
#
# --privileged lets bubblewrap make its namespaces inside Docker; on a Linux
# host the same test runs without a container once bubblewrap, socat,
# ripgrep and srt are installed.
set -eu
repo=$(cd "$(dirname "$0")/.." && pwd)
srt_version=${SRT_VERSION:-0.0.80}
exec docker run --rm --privileged -e SRT_VERSION="$srt_version" \
    -v "$repo:/src:ro" node:22-bookworm bash -euc '
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    bubblewrap socat ripgrep python3 python3-venv ca-certificates >/dev/null
npm install -g --no-fund --no-audit "@anthropic-ai/sandbox-runtime@$SRT_VERSION" >/dev/null
useradd -m probe
mkdir /work && tar -C /src --exclude=.venv --exclude=docs/_build --exclude=.git -cf - . \
    | tar -C /work -xf - && chown -R probe /work
su probe -c "
set -eu
cd /work
python3 -m venv ~/venv
~/venv/bin/pip install -q -e . pytest
echo \"bwrap \$(bwrap --version), srt \$(srt --version), \$(uname -sr)\"
~/venv/bin/python -m pytest -q -p no:cacheprovider tests/test_os_sandbox.py
"
'
