#!/usr/bin/env python3
"""Regenerate a fixed historical Gemini web-search cohort using Interactions.

Python 3.10+, standard library only. Historical response files are never edited.
Use --help for cohort selection, pilot/full execution, and saved-output validation.
"""
import argparse
import concurrent.futures
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

MODEL = "gemini-2.5-pro"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
REVISION = "2026-05-20"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_cohort(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"question_id": r["question_id"], "question": r["question"]} for r in data]
    if len(rows) != 40 or len({r["question_id"] for r in rows}) != 40:
        raise ValueError("Expected exactly 40 unique historical questions")
    if any(not re.fullmatch(r"Q\d+", r["question_id"]) or not r["question"].strip() for r in rows):
        raise ValueError("Invalid question ID or empty question")
    return rows


def payload(question, max_tokens):
    # Match the old question-only prompt and 4,000-token ceiling. Do not add a
    # request for extra references, seed, temperature, or different model.
    return {"model": MODEL, "input": question, "tools": [{"type": "google_search"}],
            "generation_config": {"max_output_tokens": max_tokens}, "store": False}


def check_key(key):
    """Read-only model metadata check; return no credential material."""
    request = Request("https://generativelanguage.googleapis.com/v1beta/models/" + MODEL,
                      headers={"x-goog-api-key": key})
    try:
        with urlopen(request, timeout=30) as response:
            data = json.load(response)
        return {"ok": True, "model": data.get("name"), "checked_at_utc": now()}
    except HTTPError as exc:
        message = exc.read(2048).decode("utf-8", errors="replace").replace(key, "[REDACTED]")
        return {"ok": False, "http_status": exc.code, "message": message, "checked_at_utc": now()}
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "checked_at_utc": now()}


def generate(question, key, max_tokens, transport="urllib"):
    if transport == "curl":
        # Supply the credential over stdin, never command-line arguments or disk.
        config = "\n".join([
            "url = " + json.dumps(ENDPOINT), 'request = "POST"',
            "header = " + json.dumps("x-goog-api-key: " + key),
            'header = "Content-Type: application/json"',
            "header = " + json.dumps("Api-Revision: " + REVISION),
            "data = " + json.dumps(json.dumps(payload(question, max_tokens))),
            'write-out = "\\n%{http_code}"'])
        result = subprocess.run(["/usr/bin/curl", "--silent", "--show-error", "--max-time", "240", "--config", "-"],
                                input=config, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"Gemini curl transport error {result.returncode}; no automatic retry")
        body, _, code = result.stdout.rpartition("\n")
        if code != "200":
            raise RuntimeError(f"Gemini HTTP {code} via curl: {body[:4096].replace(key, '[REDACTED]')}")
        return json.loads(body)
    request = Request(ENDPOINT, data=json.dumps(payload(question, max_tokens)).encode(),
                      headers={"x-goog-api-key": key, "Content-Type": "application/json",
                               "Api-Revision": REVISION}, method="POST")
    for attempt in range(3):
        try:
            with urlopen(request, timeout=240) as response:
                return json.load(response)
        except HTTPError as exc:
            message = exc.read(4096).decode("utf-8", errors="replace").replace(key, "[REDACTED]")
            if exc.code == 429 and attempt < 2:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"Gemini HTTP {exc.code} ({exc.headers.get('Content-Type', 'unknown content type')}): {message}") from None
        except (URLError, TimeoutError) as exc:
            # Do not automatically repeat an ambiguous billed POST.
            raise RuntimeError(f"Gemini transport error ({type(exc).__name__}); no automatic retry") from None


def is_redirect(url):
    parsed = urlsplit(url)
    return ((parsed.hostname or "").endswith("vertexaisearch.cloud.google.com")
            or ((parsed.hostname or "") in {"google.com", "www.google.com"}
                and parsed.path == "/url"))


def public_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Not a public HTTP(S) URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in {80, 443}:
        raise ValueError("Nonstandard URL port")
    addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("Nonpublic address blocked")


class PublicRedirects(HTTPRedirectHandler):
    def __init__(self):
        super().__init__()
        self.history = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        self.history.append({"from_url": req.full_url, "http_status": code, "to_url": newurl})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def resolve_url(url):
    result = {"provider_url": url, "checked_at_utc": now(), "provider_is_redirect": is_redirect(url)}
    redirects = PublicRedirects()
    try:
        public_url(url)
        # No model API credentials are sent to citation destinations.
        request = Request(url, headers={"User-Agent": "PanCanBench-citation-check/1.0"})
        try:
            response = build_opener(redirects).open(request, timeout=20)
        except HTTPError as exc:
            response = exc
        with response:
            final_url = response.geturl()
            result.update(http_status=response.code, final_url=final_url)
            # A destination may be identifiable despite publisher access denial;
            # record status separately and never claim HTTP 200 proves validity.
            result["resolved_original_url"] = final_url if not is_redirect(final_url) else None
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["resolved_original_url"] = None
        if redirects.history:
            destination = redirects.history[-1]["to_url"]
            result["final_url"] = destination
            if not is_redirect(destination):
                result["resolved_original_url"] = destination
                result["destination_from_redirect_only"] = True
    result["redirect_chain"] = redirects.history
    result["inline_url"] = (result.get("resolved_original_url") or url) if is_redirect(url) else url
    return result


def text_parts(raw):
    if raw.get("status") != "completed":
        raise ValueError(f"Interaction status is {raw.get('status')!r}; not accepted as a complete answer")
    returned_model = raw.get("model", "").removeprefix("models/")
    if returned_model != MODEL:
        raise ValueError(f"Unexpected response model: {returned_model!r}")
    parts = [part for step in raw.get("steps", []) if step.get("type") == "model_output"
             for part in step.get("content", []) if part.get("type") == "text"]
    if not parts or not any(p.get("text", "").strip() for p in parts):
        raise ValueError("No final answer text in Interactions steps")
    return parts


def render(parts, resolutions):
    """Insert links at UTF-8 byte boundaries; preserve all original answer text."""
    references, citations, rendered, originals = {}, [], [], []
    for part_index, part in enumerate(parts):
        original = part["text"]
        encoded = original.encode("utf-8")
        insertions = {}
        annotations = [a for a in part.get("annotations", []) if a.get("type") == "url_citation"]
        for annotation in sorted(annotations, key=lambda a: a.get("end_index", -1)):
            start, end, url = annotation.get("start_index", 0), annotation.get("end_index"), annotation.get("url")
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(encoded)
                    or not isinstance(url, str) or urlsplit(url).scheme not in {"https", "http"}):
                raise ValueError("Invalid URL citation span or URL")
            # Strict decode catches byte offsets that split a Unicode character.
            before, span, after = encoded[:end].decode(), encoded[start:end].decode(), encoded[end:].decode()
            encoded[:start].decode()
            if before and after and before[-1].isalnum() and after[0].isalnum():
                raise ValueError("Citation would split a word; inspect raw offsets before continuing")
            shown_url = resolutions[url]["inline_url"]
            number = references.setdefault(shown_url, len(references) + 1)
            insertions.setdefault(end, {})[number] = shown_url
            citations.append({"part_index": part_index, "annotation": annotation,
                              "cited_text": span, "reference_number": number, "inline_url": shown_url})
        result = encoded
        for end, sources in sorted(insertions.items(), reverse=True):
            markers = " " + " ".join(f"[{n}](<{url}>)" for n, url in sources.items())
            result = result[:end] + markers.encode() + result[end:]
        originals.append(original)
        rendered.append(result.decode())
    return {"text": "\n\n".join(originals), "text_with_inline_citations": "\n\n".join(rendered),
            "citations": citations, "unique_sources": len(references)}


def normalize(raw, row, directory, refresh_unresolved=False):
    parts = text_parts(raw)
    urls = list(dict.fromkeys(a["url"] for p in parts for a in p.get("annotations", [])
                             if a.get("type") == "url_citation" and a.get("url")))
    resolution_path = directory / "url_checks" / f"{row['question_id']}.json"
    resolutions = json.loads(resolution_path.read_text()) if resolution_path.exists() else {}
    missing = [u for u in urls if u not in resolutions or (
        refresh_unresolved and is_redirect(u) and not resolutions[u].get("resolved_original_url"))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for url, result in zip(missing, executor.map(resolve_url, missing)):
            if url in resolutions:
                previous = dict(resolutions[url])
                history = previous.pop("previous_checks", [])
                result["previous_checks"] = history + [previous]
            resolutions[url] = result
    save(resolution_path, resolutions)
    answer = render(parts, resolutions)
    warnings = []
    if not answer["citations"]:
        warnings.append("No URL citation annotations returned; answer retained without fabricating citations")
    unresolved = [u for u in urls if is_redirect(u) and not resolutions[u].get("resolved_original_url")]
    if unresolved:
        warnings.append(f"{len(unresolved)} opaque citation URLs could not be resolved")
    return {**row, "model": MODEL, **answer, "url_checks": resolutions, "warnings": warnings,
            "interaction_id": raw.get("id"), "usage": raw.get("usage", {}),
            "search_step_types": [s.get("type") for s in raw.get("steps", []) if "google_search" in s.get("type", "")]}


def export(directory, cohort, records):
    ordered = [records[r["question_id"]] for r in cohort if r["question_id"] in records]
    complete = len(ordered) == len(cohort)
    suffix = "" if complete else ".partial"
    save(directory / f"gemini_family_response_web_search_with_citations{suffix}.json",
         [{"question_id": r["question_id"], "question": r["question"],
           "responses": {MODEL: r["text_with_inline_citations"]}} for r in ordered])
    save(directory / f"citation_audit{suffix}.json", ordered)
    checks = [c for r in ordered for c in r["url_checks"].values()]
    save(directory / "run_status.json", {"updated_at_utc": now(), "model": MODEL,
         "expected_questions": len(cohort), "completed_questions": len(ordered), "complete": complete,
         "citation_annotations": sum(len(r["citations"]) for r in ordered),
         "responses_with_citation_annotations": sum(bool(r["citations"]) for r in ordered),
         "question_source_url_pairs": len(checks),
         "unique_original_urls": len({c["resolved_original_url"] for c in checks if c.get("resolved_original_url")}),
         "unresolved_opaque_url_pairs": sum(is_redirect(c["provider_url"]) and not c.get("resolved_original_url") for c in checks),
         "source_access_http_status_counts": dict(Counter(str(c.get("http_status", "access_error")) for c in checks)),
         "questions_without_annotations": [r["question_id"] for r in ordered if not r["citations"]],
         "questions_with_warnings": {r["question_id"]: r["warnings"] for r in ordered if r["warnings"]}})
    if complete:
        for name in ("gemini_family_response_web_search_with_citations.partial.json", "citation_audit.partial.json"):
            # These are only this script's superseded generated partial exports.
            (directory / name).unlink(missing_ok=True)


def read_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or not key.strip():
        raise ValueError("Export GEMINI_API_KEY before running this command")
    return key.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="Historical 40-response JSON (read only)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, dedicated run directory")
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument("--pilot-ids", nargs="+", default=["Q178", "Q102"])
    parser.add_argument("--full", action="store_true", help="Continue through all 40 after the two-question technical pilot")
    parser.add_argument("--pause-on-api-error", action="store_true", help="Keep the temporary credential in memory while an operator reviews a failed API request")
    parser.add_argument("--validate-only", action="store_true", help="Check cohort and configuration without an API call")
    parser.add_argument("--rebuild-only", action="store_true", help="Rebuild outputs from saved API responses; never generate an answer")
    parser.add_argument("--refresh-unresolved-urls", action="store_true", help="Retry only unresolved citation URLs, retaining the previous check")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(args.cohort)
    by_id = {r["question_id"]: r for r in cohort}
    if len(args.pilot_ids) != 2 or len(set(args.pilot_ids)) != 2 or any(q not in by_id for q in args.pilot_ids):
        parser.error("Choose two distinct pilot IDs from the fixed cohort")
    print(f"Validated {len(cohort)} unique questions; model={MODEL}; pilot={args.pilot_ids}", flush=True)
    if args.validate_only:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = {"cohort_source": str(args.cohort.resolve()), "cohort_sha256": digest(args.cohort),
                    "question_ids": list(by_id), "request_template": payload("<unchanged question text>", args.max_output_tokens),
                    "endpoint": ENDPOINT, "api_revision": REVISION, "pilot_ids": args.pilot_ids}
        manifest_path = args.output_dir / "run_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["protocol"] != protocol:
                raise ValueError("Existing run uses a different protocol; choose a new output directory")
            previous.setdefault("resume_history", []).append({"at_utc": now(),
                "script_sha256": digest(Path(__file__)), "full_run": args.full,
                "rebuild_only": args.rebuild_only, "refresh_unresolved_urls": args.refresh_unresolved_urls})
            save(manifest_path, previous)
        else:
            save(manifest_path, {"created_at_utc": now(), "protocol": protocol,
                 "script_sha256_at_start": digest(Path(__file__)),
                 "notes": "New generation, not a reformat of historical answers. No rubric grading or manuscript edits."})
            save(args.output_dir / "questions.json", cohort)
        key = None
        transport = "urllib"
        records = {}
        ordered_ids = args.pilot_ids + [q for q in by_id if q not in args.pilot_ids]
        for index, qid in enumerate(ordered_ids):
            if index == 2:
                if not any(records[q]["citations"] for q in args.pilot_ids):
                    raise ValueError("Neither pilot answer has citation annotations; citation pipeline cannot yet be verified")
                if any(is_redirect(u) and not check.get("resolved_original_url")
                       for q in args.pilot_ids for u, check in records[q]["url_checks"].items()):
                    raise ValueError("Technical pilot needs review: unresolved opaque URLs; stopped before full run")
                print("Technical pilot passed: complete answers and verified citation conversion. Any uncited answers are retained, not regenerated.", flush=True)
                if not args.full:
                    break
            row = by_id[qid]
            raw_path = args.output_dir / "raw" / f"{qid}.json"
            print(f"[{index + 1}/40] {qid}: {'resuming saved answer' if raw_path.exists() else 'generating'}", flush=True)
            try:
                if raw_path.exists():
                    raw = json.loads(raw_path.read_text())
                else:
                    if args.rebuild_only:
                        raise ValueError(f"No saved API response for {qid}; rebuild-only never generates answers")
                    if key is None:
                        key = read_key()
                        if not key:
                            raise ValueError("Empty Gemini API key")
                    started = time.monotonic()
                    while True:
                        try:
                            raw = generate(row["question"], key, args.max_output_tokens, transport)
                            break
                        except RuntimeError as exc:
                            save(args.output_dir / "last_error.json", {"at_utc": now(), "question_id": qid,
                                 "error": str(exc), "resolved": False})
                            if not args.pause_on_api_error:
                                raise
                            print(str(exc), flush=True)
                            action = input("API paused; key remains only in memory. Type check (read-only key check), retry, curl (same request with alternate HTTP client), or stop: ").strip().lower()
                            while action == "check":
                                check = check_key(key)
                                save(args.output_dir / "credential_check.json", check)
                                print(json.dumps(check), flush=True)
                                action = input("Type retry, curl, or stop: ").strip().lower()
                            if action == "curl":
                                transport = "curl"
                            elif action != "retry":
                                raise
                    save(raw_path, raw)
                    save(args.output_dir / "timing" / f"{qid}.json", {"saved_at_utc": now(),
                         "api_elapsed_seconds": round(time.monotonic() - started, 3), "http_transport": transport})
                records[qid] = normalize(raw, row, args.output_dir, args.refresh_unresolved_urls)
                export(args.output_dir, cohort, records)
                error_path = args.output_dir / "last_error.json"
                if error_path.exists():
                    previous_error = json.loads(error_path.read_text())
                    if previous_error.get("question_id") == qid:
                        previous_error.update(resolved=True, resolved_at_utc=now())
                        save(error_path, previous_error)
                print(f"{qid}: saved; {len(records[qid]['citations'])} inline citations, {records[qid]['unique_sources']} sources; warnings={records[qid]['warnings']}", flush=True)
            except Exception as exc:
                message = str(exc).replace(key, "[REDACTED]") if key else str(exc)
                save(args.output_dir / "last_error.json", {"at_utc": now(), "question_id": qid, "error": message})
                raise RuntimeError(message) from None
        print(f"Finished: {len(records)}/40 validated answers. Historical data unchanged.", flush=True)


if __name__ == "__main__":
    main()
