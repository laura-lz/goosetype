#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


API_ROOT = "https://api.inaturalist.org/v1"
DEFAULT_LICENSES = {"cc0", "cc-by", "cc-by-sa"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download permissively licensed goose observation photos from iNaturalist "
            "into a separate external-source folder."
        )
    )
    parser.add_argument("--taxon", default="Branta canadensis", help="Scientific or common taxon name.")
    parser.add_argument("--query", default=None, help="Optional observation search text, for example: flying.")
    parser.add_argument("--limit", type=int, default=80, help="Maximum photos to download.")
    parser.add_argument("--output", default="data/external_sources/inaturalist_geese", help="Output folder.")
    parser.add_argument("--quality-grade", default="research", help="iNaturalist quality grade filter.")
    parser.add_argument(
        "--license",
        action="append",
        default=None,
        help="Allowed lowercase photo license code. Repeat to add. Default: cc0, cc-by, cc-by-sa.",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API/download requests.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    photos_dir = output_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    allowed_licenses = set(args.license or DEFAULT_LICENSES)

    taxon_id = find_taxon_id(args.taxon)
    print(f"taxon: {args.taxon} -> {taxon_id}")

    records = []
    page = 1
    while len(records) < args.limit:
        observations = fetch_observations(
            taxon_id=taxon_id,
            query=args.query,
            quality_grade=args.quality_grade,
            page=page,
            per_page=50,
        )
        if not observations:
            break

        for observation in observations:
            for photo in observation.get("photos", []):
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
                records.append(
                    {
                        "source": "iNaturalist",
                        "taxon": args.taxon,
                        "taxon_id": taxon_id,
                        "query": args.query,
                        "observation_id": observation.get("id"),
                        "observed_on": observation.get("observed_on"),
                        "uri": observation.get("uri"),
                        "photo_id": photo.get("id"),
                        "photo_url": url,
                        "license_code": license_code,
                        "attribution": photo.get("attribution") or observation.get("user", {}).get("login"),
                        "local_path": str(path),
                    }
                )
                print(f"{len(records):04d}: {path.name} {license_code}")
                if len(records) >= args.limit:
                    break
            if len(records) >= args.limit:
                break
        page += 1
        time.sleep(args.sleep)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} photo records to {metadata_path}")


def find_taxon_id(taxon: str) -> int:
    data = request_json(f"{API_ROOT}/taxa", {"q": taxon, "per_page": 1})
    results = data.get("results", [])
    if not results:
        raise SystemExit(f"No iNaturalist taxon found for {taxon!r}")
    return int(results[0]["id"])


def fetch_observations(taxon_id: int, query: str | None, quality_grade: str, page: int, per_page: int) -> list[dict]:
    params = {
        "taxon_id": taxon_id,
        "photos": "true",
        "quality_grade": quality_grade,
        "order_by": "votes",
        "order": "desc",
        "page": page,
        "per_page": per_page,
    }
    if query:
        params["q"] = query
    data = request_json(f"{API_ROOT}/observations", params)
    return list(data.get("results", []))


def request_json(url: str, params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "GooseType/0.1 (local dataset builder)"},
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
    request = urllib.request.Request(url, headers={"User-Agent": "GooseType/0.1 (local dataset builder)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


if __name__ == "__main__":
    main()
