# Security policy

## Supported versions

Security fixes are prepared for the latest `0.1.x` release line. Development
snapshots and older pre-release builds are not supported.

## Report a vulnerability

Please use GitHub's private
[security-advisory form](https://github.com/daniiarabdiev/mlx-smolvla/security/advisories/new).
Include affected versions, impact, reproduction details, and any suggested
mitigation. Do not open a public issue until a maintainer confirms that a fix
or coordinated disclosure is ready.

This project loads model and tokenizer artifacts and can expose a trusted
pickle-based LeRobot RPC on request. Reports involving artifact integrity,
path handling, remote binding, unsafe deserialization, or command execution are
especially useful. The default server binding remains loopback-only.
