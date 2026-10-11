"""Content-addressed input snapshots and durable first-observed timestamps."""

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware prediction time is required")
    return value.astimezone(timezone.utc)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def identity(value):
    return sha256(canonical(value))


class SnapshotStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.blobs = self.directory / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.index = self.directory / "index.sqlite3"
        with closing(sqlite3.connect(self.index)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS source_snapshots (source TEXT, sha256 TEXT, first_retrieved_at TEXT, PRIMARY KEY(source,sha256))")

    def put(self, raw):
        identifier = sha256(raw)
        path = self.blobs / identifier
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=self.blobs, delete=False) as temporary:
                temporary.write(raw)
                name = temporary.name
            os.replace(name, path)
        if sha256(path.read_bytes()) != identifier:
            raise ValueError("Corrupt immutable input snapshot")
        return identifier

    def get(self, identifier):
        if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Invalid snapshot identifier")
        raw = (self.blobs / identifier).read_bytes()
        if sha256(raw) != identifier:
            raise ValueError("Corrupt immutable input snapshot")
        return raw

    def observe(self, source, raw, observed_at, metadata=None):
        identifier = self.put(raw)
        observed_at = utc(observed_at).isoformat()
        with closing(sqlite3.connect(self.index)) as db, db:
            db.execute("INSERT OR IGNORE INTO source_snapshots VALUES (?,?,?)", (source, identifier, observed_at))
            first = db.execute("SELECT first_retrieved_at FROM source_snapshots WHERE source=? AND sha256=?", (source, identifier)).fetchone()[0]
        return {**(metadata or {}), "source": source, "sha256": identifier, "first_retrieved_at": first}

    def freeze(self, payload, sources, observed_at, input_type):
        records = [self.observe(name, raw, observed_at, metadata) for name, raw, metadata in sources]
        # Split stable historical data by season. A new weather retrieval must
        # not duplicate every historical game and statistic on disk.
        parts = {}

        def part(name, value):
            record = self.observe("normalized/" + name, canonical(value), observed_at, {"schema": "normalized-component-v1"})
            records.append(record)
            parts[name] = record["sha256"]

        for year in sorted({r["season"] for r in payload["games"]}):
            part(f"games/{year}", [r for r in payload["games"] if r["season"] == year])
        for kind in ("team", "player"):
            for year in sorted({key[0] for key, _ in payload["bundle"][kind]}):
                part(f"bundle/{kind}/{year}", [r for r in payload["bundle"][kind] if r[0][0] == year])
        part("bundle/other", {k: v for k, v in payload["bundle"].items() if k not in {"team", "player", "rosters", "injuries"}})
        for kind in ("rosters", "injuries"):
            if kind in payload["bundle"]:
                part("bundle/" + kind, payload["bundle"][kind])
        plays = payload["advanced"]["plays"]
        for year in sorted({key.split("_")[0] for key in plays}):
            part("plays/" + year, {k: v for k, v in plays.items() if k.split("_")[0] == year})
        part("advanced/evidence", payload["advanced"]["evidence"])
        part("advanced/sources", payload["advanced"]["sources"])
        part("metadata", payload["metadata"])
        descriptor = {"schema": "prediction-inputs-v2", "parts": parts, "code_sha256": payload["code_sha256"]}
        normalized = self.observe("normalized-model-inputs", canonical(descriptor), observed_at, {"schema": "prediction-inputs-v2"})
        records.append(normalized)
        manifest = {"schema": "input-manifest-v1", "input_type": input_type, "sources": records,
                    "code_sha256": payload.get("code_sha256"),
                    "normalized_sha256": normalized["sha256"],
                    "available_at": max(r["first_retrieved_at"] for r in records),
                    "publication_policy": "First observed by this application, not file mtime. Historical feature replay uses the labeled next-calendar-day approximation."}
        return self.put(canonical(manifest)), manifest

    def load(self, identifier):
        manifest = json.loads(self.get(identifier))
        descriptor = json.loads(self.get(manifest["normalized_sha256"]))
        if descriptor.get("schema") != "prediction-inputs-v2":
            return descriptor, manifest  # Earlier development captures remain readable.
        payload = {"code_sha256": descriptor["code_sha256"], "games": [], "bundle": {"team": [], "player": []}, "advanced": {"plays": {}}}
        for name, blob in descriptor["parts"].items():
            value = json.loads(self.get(blob))
            components = name.split("/")
            if components[0] == "games":
                payload["games"].extend(value)
            elif components[0] == "plays":
                payload["advanced"]["plays"].update(value)
            elif name == "bundle/other":
                payload["bundle"].update(value)
            elif components[0] == "bundle" and components[1] in {"team", "player"}:
                payload["bundle"][components[1]].extend(value)
            elif len(components) == 2:
                payload[components[0]][components[1]] = value
            else:
                payload[name] = value
        payload["games"].sort(key=lambda g: (g["day"], g["kickoff"], g["id"]))
        return payload, manifest
