# Code Execution Engine (Piston) — Setup

The coding-interview feature runs candidate code in a self-hosted
[Piston](https://github.com/engineer-man/piston) container. Piston sandboxes
execution with Isolate (Linux namespaces + chroot + cgroups), so untrusted code
can't touch the host. We support **Python** and **C++**.

> Public Piston's hosted API is gated (non-commercial, by request) since
> 2026-02-15. We self-host, so this doesn't apply — just never point the backend
> at the public endpoint.

## Prerequisites

- **Docker** with a **Linux** engine. On Windows, Docker Desktop with the WSL2
  backend. Verify:
  ```
  docker info --format "{{.ServerVersion}} / {{.OSType}}"
  ```
  The OSType must print `linux` (Isolate requires Linux).

## 1. Start the Piston API container

```
docker volume create piston_packages
docker run -d --name piston_api -p 2000:2000 \
  -v piston_packages:/piston/packages \
  --privileged \
  ghcr.io/engineer-man/piston
```

- `-p 2000:2000` — HTTP API on `http://localhost:2000`
- `--privileged` — required; Isolate needs namespace/cgroup access
- the `piston_packages` volume persists installed runtimes across restarts

The container starts with **no languages installed**:
```
curl http://localhost:2000/api/v2/runtimes      # -> []
```

## 2. Install the language runtimes

List what's available, then install Python and the gcc package (which provides
C++):
```
curl -s http://localhost:2000/api/v2/packages | python -m json.tool

curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language":"python","version":"3.12.0"}'

curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language":"gcc","version":"10.2.0"}'
```

Each downloads a runtime tarball (~30-120s). Confirm:
```
curl -s http://localhost:2000/api/v2/runtimes | python -m json.tool
```
You should see `python` 3.12.0 and `c++` 10.2.0 (under the gcc runtime; its
aliases include `cpp`/`g++`).

## 3. Verify execution

```
# Python -> run.stdout "4\n", run.code 0
curl -s -X POST http://localhost:2000/api/v2/execute \
  -H "Content-Type: application/json" \
  -d '{"language":"python","version":"3.12.0","files":[{"content":"print(2+2)"}]}'

# C++ -> compile.code 0, run.stdout "4"
curl -s -X POST http://localhost:2000/api/v2/execute \
  -H "Content-Type: application/json" \
  -d '{"language":"c++","version":"10.2.0","files":[{"name":"main.cpp","content":"#include <iostream>\nint main(){std::cout<<2+2;}"}]}'
```

> Windows note: the JSON-in-single-quotes form above is for bash/git-bash. In
> PowerShell, build the body with a hashtable + `ConvertTo-Json` and
> `Invoke-RestMethod` instead, to avoid quote-escaping pain.

## API contract (used by `executor.py`)

`POST /api/v2/execute`
```json
{
  "language": "python", "version": "3.12.0",
  "files": [{ "name": "main.cpp", "content": "..." }],
  "stdin": "", "run_timeout": 3000, "compile_timeout": 10000
}
```
Response has separate `compile` and `run` objects, each with `stdout`,
`stderr`, and `code`. For Python only `run` matters; for C++ a build failure
shows up in `compile.stderr` with a non-zero `compile.code`. C++ files need a
`name` ending in `.cpp`.

## Lifecycle

```
docker stop piston_api        # stop
docker start piston_api       # restart (runtimes persist via the volume)
docker rm -f piston_api       # remove (volume survives; recreate to reuse runtimes)
```
