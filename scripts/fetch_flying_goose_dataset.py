#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path


API_ROOT = "https://api.inaturalist.org/v1"
DEFAULT_LICENSES = {"cc0", "cc-by", "cc-by-sa"}
DEFAULT_TAXA = (
    "Branta canadensis",
    "Anser anser",
    "Anser albifrons",
    "Branta bernicla",
    "Branta hutchinsii",
    "Branta leucopsis",
    "Anser caerulescens",
)
DEFAULT_QUERIES = (
    "flying",
    "flight",
    "in flight",
    "landing",
    "taking off",
    "flock flying",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Curate permissively licensed flying-goose source photos from iNaturalist. "
            "The output is source photos plus metadata; run extract_instances_v2.py afterward."
        )
    )
    parser.add_argument("--output", default="data/external_sources/inaturalist_curated_flying_geese")
    parser.add_argument("--limit", type=int, default=360, help="Maximum total photos to download.")
    parser.add_argument("--per-taxon-query", type=int, default=18, help="Max photos per taxon/query pair.")
    parser.add_argument("--quality-grade", default="research")
    parser.add_argument("--taxon", action="append", default=None, help="Repeat to override the default goose taxa.")
    parser.add_argument("--query", action="append", default=None, help="Repeat to override default flight terms.")
    parser.add_argument(
        "--license",
        action="append",
        default=None,
        help="Allowed lowercase photo license code. Repeat to add. Default: cc0, cc-by, cc-by-sa.",
    )
    parser.add_argument("--sleep", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = Path(args.output)
    photos_dir = output_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    allowed_licenses = set(args.license or DEFAULT_LICENSES)
    taxa = tuple(args.taxon or DEFAULT_TAXA)
    queries = tuple(args.query or DEFAULT_QUERIES)

    metadata_path = output_dir / "metadata.json"
    records = load_existing_records(metadata_path)
    seen_photo_ids = {str(record.get("photo_id") or "") for record in records}
    seen_observation_ids = {str(record.get("observation_id") or "") for record in records}
    print(f"resume: {len(records)} existing records", flush=True)
    taxa_cache: dict[str, int] = {}

    for taxon in taxa:
        taxon_id = taxa_cache.setdefault(taxon, find_taxon_id(taxon))
        for query in queries:
            if len(records) >= args.limit:
                break
            print(f"search: taxon={taxon} query={query!r}", flush=True)
            pair_count = 0
            page = 1
            while len(records) < args.limit and pair_count < args.per_taxon_query:
                try:
                    observations = fetch_observations(
                        taxon_id=taxon_id,
                        query=query,
                        quality_grade=args.quality_grade,
                        page=page,
                        per_page=50,
                    )
                except (TimeoutError, OSError, socket.timeout) as error:
                    print(f"warn: skipping page after API error: {error}", flush=True)
                    break
                if not observations:
                    break
                for observation in observations:
                    observation_id = str(observation.get("id") or "")
                    for photo in observation.get("photos", []):
                        photo_id = str(photo.get("id") or "")
                        if not photo_id or photo_id in seen_photo_ids:
                            continue
                        license_code = str(photo.get("license_code") or "").lower()
                        if license_code not in allowed_licenses:
                            continue
                        url = best_photo_url(photo)
                        if not url:
                            continue
                        filename = build_filename(observation, photo, url)
                        path = photos_dir / filename
                        if not path.exists():
                            download(url, path)
                            time.sleep(args.sleep)
                        seen_photo_ids.add(photo_id)
                        seen_observation_ids.add(observation_id)
                        record = {
                            "source": "iNaturalist",
                            "dataset_role": "flying_goose_source",
                            "pose_query": query,
                            "taxon": taxon,
                            "taxon_id": taxon_id,
                            "observation_id": observation.get("id"),
                            "observed_on": observation.get("observed_on"),
                            "uri": observation.get("uri"),
                            "photo_id": photo.get("id"),
                            "photo_url": url,
                            "license_code": license_code,
                            "attribution": photo.get("attribution") or observation.get("user", {}).get("login"),
                            "local_path": str(path),
                        }
                        records.append(record)
                        pair_count += 1
                        write_metadata(metadata_path, allowed_licenses, taxa, queries, records, seen_observation_ids)
                        print(f"{len(records):04d}: {path.name} {taxon} {query!r} {license_code}", flush=True)
                        if len(records) >= args.limit or pair_count >= args.per_taxon_query:
                            break
                    if len(records) >= args.limit or pair_count >= args.per_taxon_query:
                        break
                page += 1
                time.sleep(args.sleep)
        if len(records) >= args.limit:
            break

    write_metadata(metadata_path, allowed_licenses, taxa, queries, records, seen_observation_ids)
    print(f"Wrote {len(records)} photo records to {metadata_path}", flush=True)


def load_existing_records(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        return []
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("records", []))
    if isinstance(data, list):
        return data
    return []


def write_metadata(
    metadata_path: Path,
    allowed_licenses: set[str],
    taxa: tuple[str, ...],
    queries: tuple[str, ...],
    records: list[dict],
    seen_observation_ids: set[str],
) -> None:
    metadata = {
        "source": "iNaturalist",
        "description": "Permissively licensed goose photos curated with flying-related search terms.",
        "licenses": sorted(allowed_licenses),
        "taxa": list(taxa),
        "queries": list(queries),
        "photo_count": len(records),
        "observation_count": len(seen_observation_ids),
        "records": records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def find_taxon_id(taxon: str) -> int:
    data = request_json(f"{API_ROOT}/taxa", {"q": taxon, "per_page": 1})
    results = data.get("results", [])
    if not results:
        raise SystemExit(f"No iNaturalist taxon found for {taxon!r}")
    return int(results[0]["id"])


def fetch_observations(taxon_id: int, query: str, quality_grade: str, page: int, per_page: int) -> list[dict]:
    params = {
        "taxon_id": taxon_id,
        "photos": "true",
        "quality_grade": quality_grade,
        "q": query,
        "order_by": "votes",
        "order": "desc",
        "page": page,
        "per_page": per_page,
    }
    data = request_json(f"{API_ROOT}/observations", params)
    return list(data.get("results", []))


def request_json(url: str, params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "GooseType/0.1 (local flying dataset builder)"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def best_photo_url(photo: dict) -> str | None:
    url = photo.get("url") or photo.get("large_url") or photo.get("medium_url")
    if not url:
        return None
    return str(url).replace("square.", "large.").replace("medium.", "large.")


def build_filename(observation: dict, photo: dict, url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".jpg"
    return f"inat_{observation.get('id')}_{photo.get('id')}{suffix}"


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "GooseType/0.1 (local flying dataset builder)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


if __name__ == "__main__":
    main()
