"""Persistenta rezultatelor: o linie CSV per (task, model, repetition).

Fisierul se deschide in append, cu antet scris o singura data. Coloanele sunt
fixe si ordonate, ca sa poata fi incarcat direct in pandas fara curatare.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import threading
from datetime import datetime, timezone

from app import config

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR = os.path.join(_ROOT, "benchmark", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "metrics.csv")
# Fisier separat, curat de rulari brute: aici ajung doar observatiile pe care
# cercetatorul le confirma explicit din UI (verdict + note). `metrics.csv`
# ramane jurnalul complet, `observatii.csv` devine setul adnotat pe care se
# scrie analiza calitativa si taxonomia de erori.
OBSERVATIONS_CSV = os.path.join(RESULTS_DIR, "observatii.csv")

# Scrierile pot veni din cereri HTTP concurente; antetul trebuie scris o
# singura data, iar liniile nu trebuie sa se intercaleze.
_lock = threading.Lock()

COLUMNS = [
    # identificare
    "ts", "run_id", "task_id", "difficulty", "title", "servers",
    "model_id", "model_label", "repetition",
    # Which description set the tools were published with. The descriptions are
    # the apparatus, so this is an independent variable: rows from the two arms
    # must never be averaged together.
    "descriptions_regime", "descriptions_chars",
    # The two guidance layers that are no longer bound to the regime. Recorded
    # so a file can be checked, rather than trusted, to have held them
    # constant: every row of one campaign must carry the same prompt_sha.
    "prompt_regime", "prompt_sha", "guidance_runtime",
    # rezultat
    "strict_success", "partial_credit",
    "auto_assertions_passed", "auto_assertions_total", "manual_assertions",
    # Two readings of the same run, so that agreement between them can be
    # reported rather than assumed. `manual_verdict` belongs to the primary
    # evaluator (ENVIROO_EVALUATOR_PRIMARY); any other reader writes the
    # second. Who wrote a verdict is not recoverable from these columns alone,
    # so observatii.csv is the authoritative record: it is keyed by
    # (run_id, evaluator) and holds every reader's verdict with its own note.
    "manual_verdict", "manual_verdict_2",
    # decizii (M1-M5)
    "m1_tool_precision", "m2_tool_coverage", "m3_tool_f1",
    "m4_param_accuracy", "m4_param_ok_calls",
    "m5_dependency_violations", "dependencies_evaluated",
    # eficienta (M6-M11)
    "golden_steps", "calls_total", "calls_useful", "calls_discovery",
    "calls_redundant", "m6_redundancy", "m8_semantic_retry",
    "m9_call_error_rate", "m10_recovery", "m11_rounds",
    # economie (M12-M16)
    "m12_prompt_tokens", "m12_completion_tokens", "m12_total_tokens",
    "m13_context_overhead", "m14_cost_usd", "m16_duration_ms",
    # siguranta (M21-M22)
    "m21_fabrication", "m22_hallucination", "empty_answer", "reasoning_chars",
    # siguranta la nivel de afirmatie (M23-M25)
    # M21 se declanseaza doar cand niciun tool nu a fost apelat, deci nu poate
    # vedea un numar inventat intr-o rulare care i-a apelat corect. Coloanele
    # de aici stau LANGA ea, nu in locul ei, iar `strict_success_grounded`
    # e succesul strict cu fabricarea de afirmatii luata in calcul.
    "m23_claim_fabrication", "m23_numbers_total", "m23_numbers_ungrounded",
    "m23_grounding_ratio", "m24_ids_ungrounded", "m25_status_contradicted",
    "strict_success_grounded",
    # provenienta: ce a servit cererea, si cu ce setari de eșantionare. A cere
    # `gpt-5` nu spune nimic despre snapshot-ul datat care a answer, iar un
    # model open-weights ajunge la alt upstream la fiecare cerere.
    "model_route", "model_resolved", "provider_upstream",
    "temperature", "top_p", "max_output_tokens", "token_param",
    "tool_choice", "max_rounds",
    # diagnostic
    # The claims the grounding indicator could not match, verbatim. Without
    # them a flagged run can only be rechecked by calling the model again, and
    # an indicator nobody can audit cannot be reported.
    "grounding_details",
    "tools_called", "param_errors", "run_error", "answer", "note",
]


# Un answer de model contine linii noi. CSV le permite in interiorul unui camp
# ghilimelat, dar Excel NU le pastreaza la salvare: rupe inregistrarea in cate
# un rand pentru fiecare linie fizica si completeaza restul cu virgule, iar
# fisierul devine ilizibil si pentru cod, si pentru om. Cum acest CSV e deschis
# in Excel ca parte normala a analizei, orice camp trebuie sa incapa pe UN
# SINGUR rand. Marcatorul e reversibil: `unescape_cell` il transforma inapoi.
_NEWLINE_MARK = " \\n "


def escape_cell(text: str) -> str:
    """Textul, garantat pe o singura linie."""
    out = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return out.replace("\n", _NEWLINE_MARK).strip()


def unescape_cell(text: str) -> str:
    """Inversul lui `escape_cell`, pentru cine reciteste raspunsurile din CSV."""
    return str(text).replace(_NEWLINE_MARK, "\n").replace("\\n", "\n")


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return escape_cell(json.dumps(value, ensure_ascii=False))
    return escape_cell(value)


def _has_rows(path: str) -> bool:
    """True cand fisierul contine deja cel putin o linie CSV reala.

    Dimensiunea nu e testul potrivit. Un fisier golit in Excel, sau intr-un
    editor care ii pastreaza codificarea, ramane cu BOM-ul de 3 octeti si
    uneori cu un newline in plus; os.path.getsize raporteaza atunci un
    continut care nu exista, header-ul nu se mai scrie niciodata, si toti
    cititorii din aval (csv.DictReader) iau primul rand de DATE drept nume
    de coloane — read_results, next_repetition si set_manual_verdict includ.
    """
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            return any(line.strip() for line in fh)
    except OSError:
        return False


def _migrate_header_if_needed() -> bool:
    """Rescrie fisierul daca antetul lui nu mai e COLUMNS. Intoarce True daca a facut-o.

    COLUMNS creste pe masura ce apar metrici noi, dar antetul se scria o
    singura data, la crearea fisierului. Un rand nou cu 49 de campuri scris
    sub un antet de 47 nu da nicio eroare: se decaleaza tacit toate coloanele
    de dupa punctul de insertie, iar valorile ajung sub alte nume. S-a
    intamplat de doua ori inainte ca verificarea asta sa existe.

    Randurile vechi se citesc cu numele lor de atunci si se rescriu in ordinea
    noua; coloanele care nu existau raman goale, ceea ce e adevarat.
    """
    if not _has_rows(RESULTS_CSV):
        return False
    with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0] == COLUMNS:
        return False

    old_names = rows[0]
    migrated = [dict(zip(old_names, r)) for r in rows[1:] if any(r)]
    shutil.copy(RESULTS_CSV, RESULTS_CSV + ".pre_migration.bak")
    with open(RESULTS_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerows([[d.get(c, "") for c in COLUMNS] for d in migrated])
    return True


def append_result(row: dict) -> str:
    """Adauga o linie in metrics.csv. Intoarce calea fisierului."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with _lock:
        _migrate_header_if_needed()
        fresh = not _has_rows(RESULTS_CSV)
        # "w" cand fisierul e gol: trunchierea inlatura un BOM sau un rand
        # gol ramas in urma, care altfel ar sta deasupra header-ului pe toata
        # durata de viata a fisierului.
        # utf-8-sig: Excel pe Windows deschide altfel diacriticele gresit.
        with open(RESULTS_CSV, "w" if fresh else "a",
                  encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            if fresh:
                writer.writerow(COLUMNS)
            writer.writerow([_fmt(row.get(col)) for col in COLUMNS])
    return RESULTS_CSV


def next_repetition(task_id: str, model_id: str, regim: str = "rich") -> int:
    """Cate rulari exista deja pentru aceasta pereche SI acest regim, +1.

    Regimul intra in cheie pentru ca cele doua seturi de descrieri sunt brate
    diferite ale aceluiasi experiment: numarate impreuna, repetarile ar sugera
    trei rulari ale aceleiasi conditii acolo unde sunt doua conditii diferite.
    Randurile scrise inainte ca aceasta coloana sa existe se citesc ca `rich`,
    care e ce erau.
    """
    if not os.path.exists(RESULTS_CSV):
        return 1
    count = 0
    try:
        with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if (row.get("task_id") == task_id and row.get("model_id") == model_id
                        and (row.get("descriptions_regime") or "rich") == regim):
                    count += 1
    except Exception:
        return 1
    return count + 1


OBSERVATION_COLUMNS = [
    "ts", "run_id", "task_id", "difficulty", "title", "model_id", "model_label",
    "repetition", "evaluator", "manual_verdict", "note",
    "strict_success", "partial_credit",
    "m1_tool_precision", "m2_tool_coverage", "m3_tool_f1", "m4_param_accuracy",
    "m5_dependency_violations", "golden_steps", "calls_total", "calls_redundant",
    "m8_semantic_retry", "m9_call_error_rate", "m11_rounds",
    "m12_total_tokens", "m13_context_overhead", "m14_cost_usd", "m16_duration_ms",
    "m21_fabrication", "m22_hallucination", "empty_answer", "reasoning_chars",
    "tools_called", "param_errors", "answer",
]


def find_result(run_id: str) -> dict | None:
    """Linia din metrics.csv pentru o rulare. Observatia se compune din
    datele deja masurate, nu din ce trimite clientul — altfel adnotarea ar
    putea contrazice masuratoarea."""
    if not os.path.exists(RESULTS_CSV):
        return None
    with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("run_id") == run_id:
                return row
    return None


def append_observation(run_id: str, verdict: str, note: str,
                       evaluator: str | None = None) -> str | None:
    """Adauga o observatie adnotata in observatii.csv.

    Cheia e (run_id, evaluator), nu run_id singur. Acelasi cititor care revine
    asupra unui verdict il rescrie, ca inainte; un AL DOILEA cititor al
    aceleiasi rulari se adauga langa primul, in loc sa il inlocuiasca. Fara
    asta, acordul dintre doi evaluatori nu poate fi calculat, pentru ca al
    doilea verdict stergea primul - exact ce semnaleaza secțiunea 6.5.
    """
    source = find_result(run_id)
    if source is None:
        return None
    evaluator = (evaluator or config.EVALUATOR).strip() or config.EVALUATOR
    row = {col: source.get(col, "") for col in OBSERVATION_COLUMNS}
    row.update(ts=now_iso(), run_id=run_id, evaluator=evaluator,
               manual_verdict=verdict, note=note)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with _lock:
        existing: list[dict] = []
        if os.path.exists(OBSERVATIONS_CSV) and os.path.getsize(OBSERVATIONS_CSV) > 0:
            with open(OBSERVATIONS_CSV, encoding="utf-8-sig", newline="") as fh:
                existing = [r for r in csv.DictReader(fh)
                            if not (r.get("run_id") == run_id
                                    and (r.get("evaluator") or config.EVALUATOR_PRIMARY)
                                    == evaluator)]
        existing.append(row)
        with open(OBSERVATIONS_CSV, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OBSERVATION_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing)
    return OBSERVATIONS_CSV


def count_observations() -> int:
    if not os.path.exists(OBSERVATIONS_CSV):
        return 0
    with open(OBSERVATIONS_CSV, encoding="utf-8-sig", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def set_manual_verdict(run_id: str, verdict: str, note: str = "",
                       evaluator: str | None = None) -> bool:
    """Completeaza verdictul manual pentru o rulare deja scrisa.

    Aserțiunile de tip `consistenta`, `clasa_dominanta` sau `camp_nenul` nu pot
    fi evaluate mecanic; cercetatorul le marcheaza din UI dupa ce citeste
    raspunsul. Rescrie fisierul in intregime — la ordinul de marime al acestui
    experiment (sute de linii) e mai simplu si mai sigur decat editarea pe loc.
    """
    if not os.path.exists(RESULTS_CSV):
        return False
    with _lock:
        with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        found = False
        for row in rows:
            if row.get("run_id") == run_id:
                # Each reader writes their own column, so a second
                # reading of the same run sits beside the first instead
                # of replacing it. observatii.csv holds both in full.
                who = (evaluator or config.EVALUATOR).strip() or config.EVALUATOR
                column = ("manual_verdict" if who == config.EVALUATOR_PRIMARY
                          else "manual_verdict_2")
                row[column] = verdict
                if note:
                    row["note"] = note
                found = True
        if not found:
            return False
        with open(RESULTS_CSV, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return True


def read_results(limit: int | None = None) -> list[dict]:
    """Ultimele rezultate, cele mai recente primele."""
    if not os.path.exists(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.reverse()
    return rows[:limit] if limit else rows


def summary_by_model() -> list[dict]:
    """Agregat per model: rate de succes si medii, pentru tabelul din UI."""
    rows = read_results()
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row.get("model_id", "?"), []).append(row)

    def _mean(values: list[str]) -> float | None:
        nums = []
        for v in values:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                continue
        return sum(nums) / len(nums) if nums else None

    out = []
    for model_id, group in sorted(buckets.items()):
        out.append({
            "model_id": model_id,
            "rulari": len(group),
            "succes": _mean([r.get("strict_success", "") for r in group]),
            "f1_tool": _mean([r.get("m3_tool_f1", "") for r in group]),
            "param": _mean([r.get("m4_param_accuracy", "") for r in group]),
            "redundanta": _mean([r.get("m6_redundancy", "") for r in group]),
            "retry": _mean([r.get("m8_semantic_retry", "") for r in group]),
            "tokens": _mean([r.get("m12_total_tokens", "") for r in group]),
            "cost": _mean([r.get("m14_cost_usd", "") for r in group]),
            "durata_ms": _mean([r.get("m16_duration_ms", "") for r in group]),
        })
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
