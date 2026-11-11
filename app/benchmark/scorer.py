"""Motorul de metrici M1-M22.

Primeste o rulare (apelurile de tool-uri, raspunsul final, consumul de tokens)
si taskul din `golden_chains.json`, si intoarce metricile.

Principii:
- Unitatea principala de analiza este APELUL de tool, nu taskul: la 22 de
  taskuri intervalul de incredere e prea larg pentru afirmatii fine, dar la
  ~88 de apeluri per trecere devine utilizabil.
- Selectia tool-ului (M1-M3) si legarea argumentelor (M4) se raporteaza
  SEPARAT: sunt esecuri diferite, cu remedii diferite (rutare vs. descrieri
  de tool mai bune). Amestecate, se pierde exact concluzia utila.
- Metricile de eficienta se calculeaza doar pe rulari care au produs apeluri;
  altfel abandonul imediat ar aparea ca eficienta maxima.
- Aserțiunile care nu pot fi evaluate mecanic sunt marcate `manual` si lasate
  cercetatorului, nu ghicite.
"""
from __future__ import annotations

import itertools
import json
import re
import unicodedata
from collections import Counter
from typing import Any

# ENVIROO has no discovery tools: the tools of the shelter server are
# all substantive, and there is no schema a model must inspect before it can
# ask a question. The mechanism is kept (the scorer is shared with the GeoWeb
# harness, where table and column names cannot be guessed) but the set is
# empty, so here every call counts either as useful or as redundant.
DISCOVERY_TOOLS: set[str] = set()

# Chei care denumesc o latitudine, respectiv o longitudine. Folosite pentru a
# detecta specific inversarea lat/lon — cea mai frecventa eroare de parametru
# din tot sistemul, pentru ca serverul GIS lucreaza in (lat, lon) iar ORS in
# [lon, lat].
_LAT_KEYS = {"lat", "latitude", "start_lat", "end_lat"}
_LON_KEYS = {"lon", "lng", "longitude", "start_lon", "end_lon"}

_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

# Digits that are not measurements. Both of these inject numbers into an answer
# that the assertions then read as reported values: `S01545` contributes 1545,
# and `Str. Garii nr. 19` contributes 19 — which is how an interval of 8 to 20
# minutes came to be satisfied by a house number, on a run that never measured
# a walking time at all.
_IDENTIFIER_RE = re.compile(r"\bS\d{4,6}\b", re.IGNORECASE)
_HOUSE_NUMBER_RE = re.compile(r"\b(?:nr|no)\.?\s*\d+(?:\s*[-/]\s*\d+)?",
                              re.IGNORECASE)

# `25,792` is one number and it is twenty-five thousand, not 25.792. The rule is
# the group of exactly three digits: a separator followed by three digits, in
# runs, is a thousands separator in every convention these answers use, while a
# comma followed by one or two digits stays the decimal comma a model writing
# Romanian uses for 2,8 minutes.
#
# The dot is deliberately NOT treated this way. `23.801` is a thousands
# separator in Romanian and a decimal point in English, and the same answers
# carry coordinates and walking times written with a decimal point, so
# reinterpreting it would break far more than it fixed.
_THOUSANDS_RE = re.compile(
    r"(?<![\d.,])(\d{1,3})((?:[,\u00a0\u202f ]\d{3})+)(?![\d.,]*\d)")


def _strip_thousands(match: re.Match) -> str:
    return match.group(1) + re.sub(r"\D", "", match.group(2))

# Parametri liber-formulabili: exista o infinitate de forme corecte (un SELECT
# se poate scrie in zeci de feluri echivalente), deci compararea literala ar
# produce doar fals-pozitive si ar deprima M4 pentru toate modelele. Se verifica
# doar ca modelul a completat parametrul; corectitudinea rezultatului e prinsa
# de asertiuni, care sunt instrumentul potrivit pentru asa ceva.
_FREEFORM_PARAMS = {"sql"}

# Formulari prin care un model semnaleaza corect ca nu a reusit. Prezenta lor
# distinge "a esuat si a spus" de "a esuat si a inventat". Promptrurile
# benchmark-ului sunt in engleza (v2.0), dar marker-ele romanesti raman pentru
# compatibilitate cu rulari mai vechi si cu modele care raspund in romana.
_FAILURE_MARKERS = (
    "nu am reusit", "nu am reușit", "nu pot", "eroare", "a esuat", "a eșuat",
    "nu s-a putut", "indisponibil", "nu am putut", "esuat", "eșuat",
    "nu am obtinut", "nu am obținut",
    "failed", "error", "could not", "couldn't", "unable to", "was not able",
    "did not succeed", "unavailable", "no result", "cannot",
    # A tool that refuses an ambiguous or impossible request has NOT been
    # hidden by the model when the model passes the refusal on. These are the
    # words it passes it on with, and without them a correct report of a
    # refusal was being counted as a concealed failure (M22) — which is how
    # C2, C3 and C4 came to be flagged for doing exactly the right thing.
    "ambiguous", "ambiguity", "more than one", "multiple matches",
    "several matches", "not specific enough", "need more", "please specify",
    "which city", "does not have", "there is no", "no such", "not found",
    "not recorded", "not available", "no data", "does not exist",
    "ambiguu", "mai multe rezultate", "mai multe potriviri", "nu este clar",
    "nu exista", "nu există", "nu are", "nu s-a gasit", "nu s-a găsit",
    "precizati", "precizați",
)


# ── Normalizari ────────────────────────────────────────────────────────────

def _strip_diacritics(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


# Typographic space the models emit where a plain space is meant: a
# non-breaking space inside "Cluj Napoca" or "8 minutes" is not a different
# word, and a phrase table written with plain spaces would never match it.
_UNICODE_SPACE = re.compile(r"[    ⁠]")


def _norm_text(text: str) -> str:
    """Casefold + fara diacritice: 'Funcțional' si 'functional' se potrivesc."""
    return _UNICODE_SPACE.sub(" ", _strip_diacritics(str(text)).casefold())


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", "."))
        except ValueError:
            return None
    return None


def _numbers_in(text: str) -> list[tuple[float, int]]:
    """Numerele din text, fiecare cu numarul lui de zecimale.

    Zecimalele conteaza: modelele raporteaza in proza valori rotunjite
    ('44.9277' pentru 44.927727). Fara aceasta informatie, graderul ar
    respinge un answer corect doar pentru ca e mai putin precis decat
    iesirea bruta a tool-ului.

    Inainte de extragere, textul e curatat de cifrele care nu sunt masuratori:
    identificatorii de adapost si numerele de imobil. Si separatorul de mii e
    lipit inapoi, ca `25,792` sa fie un numar, nu doua si nici 25.792.
    """
    cleaned = _IDENTIFIER_RE.sub(" ", text or "")
    cleaned = _HOUSE_NUMBER_RE.sub(" ", cleaned)
    cleaned = _THOUSANDS_RE.sub(_strip_thousands, cleaned)

    out = []
    for match in _NUM_RE.findall(cleaned):
        normalized = match.replace(",", ".")
        try:
            value = float(normalized)
        except ValueError:
            continue
        decimals = len(normalized.split(".")[1]) if "." in normalized else 0
        out.append((value, decimals))
    return out


# Sub 2 zecimale, "corect rotunjit" devine prea permisiv (la latitudine, o
# zecimala inseamna ~11 km), deci rotunjirea se accepta doar de aici in sus.
_MIN_ROUND_DECIMALS = 2


def _matches_number(reported: float, decimals: int, target: float, tol: float) -> bool:
    if abs(reported - target) <= tol:
        return True
    # Valoare corect rotunjita de model: 44.9277 pentru 44.927727.
    return decimals >= _MIN_ROUND_DECIMALS and round(target, decimals) == reported


# Parametri care numesc un loc in cuvinte. Un geocoder citeste continutul, nu
# forma: `Strada Horea 27, Cluj-Napoca` si `Str. Horea nr. 27, Cluj` sunt
# aceeasi cerere. Comparate literal, a doua forma ar fi punctata ca eroare de
# parametru desi rezolva identic — iar `arg_alternatives` ar trebui sa enumere
# la nesfarsit forme echivalente.
_ADDRESS_PARAMS = {"query"}

# Cuvintele care nu poarta informatie intr-o adresa: tipul arterei, marcajul de
# numar, tara. `cluj` si `napoca` intra aici pentru ca `Cluj`, `Cluj-Napoca` si
# o adresa fara oras denumesc acelasi loc in toate taskurile: adaugarea sau
# omiterea orasului e o alegere de interogare, nu o eroare de parametru, si
# scorul ei o numara de doua ori daca doar una din cele doua jumatati ale
# numelui e ignorata. Echivalenta geografica a rezultatului e verificata de
# aserțiunea manuala a taskului, care e instrumentul potrivit pentru ea. Ce ramane sunt numele si numerele — inclusiv
# `bloc`, `scara` si `etaj`, care NU sunt zgomot ignorabil: prezenta lor in
# interogare e exact esecul pe care A1 il masoara.
_ADDRESS_STOPWORDS = {
    "str", "strada", "stradela", "bd", "bdul", "b", "dul", "bulevardul",
    "blvd", "boulevard", "street", "st", "calea", "piata", "square",
    "nr", "no", "number", "romania", "ro", "cluj", "napoca", "judetul", "judet",
    "county", "the", "of", "din", "la", "de",
}

_ADDRESS_SPLIT_RE = re.compile(r"[^0-9a-z]+")


def _address_tokens(value: Any) -> frozenset[str]:
    """Cuvintele purtatoare de sens dintr-o adresa, fara diacritice si fara
    ordine. Ordinea nu conteaza: `Cluj-Napoca, Strada Horea 27` si `Strada
    Horea 27, Cluj-Napoca` sunt aceeasi adresa pentru orice geocoder."""
    tokens = _ADDRESS_SPLIT_RE.split(_norm_text(value))
    return frozenset(t for t in tokens if t and t not in _ADDRESS_STOPWORDS)


def _addresses_equal(expected: Any, actual: Any) -> bool:
    return bool(_address_tokens(expected)) and _address_tokens(expected) == _address_tokens(actual)


# Un numar trimis ca `"46,7712"`. Tool-ul il refuza, deci a-l socoti corect ar
# ascunde o eroare reala de parametru in spatele tolerantei graderului: exact
# esecul pe care P1 il provoaca intentionat.
_COMMA_DECIMAL_RE = re.compile(r"^[-+]?\d+,\d+$")


def _is_comma_decimal(expected: Any, received: Any) -> bool:
    return (_as_number(expected) is not None
            and isinstance(received, str)
            and bool(_COMMA_DECIMAL_RE.match(received.strip())))


def _is_placeholder(value: Any) -> bool:
    """`$P1.latitude` sau `<task_geometry>` — valori care depind de rulare.

    Nu pot fi comparate literal, deci se verifica doar prezenta parametrului.
    """
    return isinstance(value, str) and (value.startswith("$P") or value.startswith("<"))


def _values_equal(expected: Any, actual: Any, rel_tol: float = 1e-4) -> bool:
    exp_num, act_num = _as_number(expected), _as_number(actual)
    if exp_num is not None and act_num is not None:
        tol = max(abs(exp_num) * rel_tol, 1e-6)
        return abs(exp_num - act_num) <= tol
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return False
        return all(_values_equal(v, actual[k], rel_tol) for k, v in expected.items())
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(_values_equal(e, a, rel_tol) for e, a in zip(expected, actual))
    return _norm_text(expected) == _norm_text(actual)


# ── Potrivirea apelurilor cu lantul de aur ─────────────────────────────────

def _candidate_calls(tool: str, calls: list[dict], used: set[int]) -> list[int]:
    """Unused calls to `tool`, successful ones first, then in call order.

    The precedence is the one the greedy matcher had: a model that calls a tool
    with a malformed argument, is refused and retries correctly is scored on the
    corrected call. Recovering from an error is a competence under measurement,
    so scoring the recovered call as a parameter error would invert it.
    """
    return sorted(
        (ai for ai, c in enumerate(calls)
         if ai not in used and c.get("tool") == tool),
        key=lambda ai: (not calls[ai].get("ok", True), ai),
    )


_MAX_PERMUTED_CANDIDATES = 6


def _assignment_score(gis: list[int], perm: tuple[int, ...],
                      golden: list[dict], calls: list[dict]) -> tuple[int, int]:
    """How well one assignment of golden steps to calls binds its arguments.

    Correct decisive arguments dominate; errors subtract; a successful call
    breaks what is left of the tie. Used only to choose between assignments,
    never reported.
    """
    correct = errors = ok_calls = 0
    for gi, ai in zip(gis, perm):
        cmp = _compare_args(golden[gi].get("args", {}), calls[ai].get("args"),
                            golden[gi].get("arg_alternatives"))
        correct += cmp["corecti"]
        errors += len(cmp["erori"])
        ok_calls += 1 if calls[ai].get("ok", True) else 0
    return (correct - errors, ok_calls)


def _match_chain(golden: list[dict], calls: list[dict]) -> dict[int, int]:
    """Pair every step of the reference chain with the call that carried it out.

    Steps are grouped by tool, because only calls to the same tool can compete
    for the same step. Within a group of one the pairing is the first unused
    call, as before. Within a group of several it is the assignment that binds
    the most arguments correctly, NOT the order the chain happens to list them
    in: when a task routes two tied shelters, or geocodes two addresses, the
    order between those steps is not part of what the task asks. Scoring it
    turned a model that answered B3 correctly in the other order into two
    wrong-value errors - an artefact of the chain, not a mistake of the model.

    A model that calls the right tool more often than the chain needs consumes
    one pairing per step; the rest become redundant, which is the intended
    reading - redundancy is a cost, not an error.
    """
    matches: dict[int, int] = {}
    used: set[int] = set()

    groups: dict[Any, list[int]] = {}
    for gi, step in enumerate(golden):
        groups.setdefault(step.get("tool"), []).append(gi)

    # Smaller groups first, so an unambiguous single step is never outbid for a
    # call by a group that has alternatives left.
    for tool, gis in sorted(groups.items(), key=lambda kv: len(kv[1])):
        cands = _candidate_calls(tool, calls, used)
        if not cands:
            continue
        if len(gis) == 1:
            matches[gis[0]] = cands[0]
            used.add(cands[0])
            continue
        # Bounded: the search is over the best few candidates, so a model that
        # called one tool a dozen times cannot make scoring factorial.
        pool = cands[:_MAX_PERMUTED_CANDIDATES]
        k = min(len(gis), len(pool))
        best = max(itertools.permutations(pool, k),
                   key=lambda perm: _assignment_score(gis[:k], perm, golden, calls))
        for gi, ai in zip(gis[:k], best):
            matches[gi] = ai
            used.add(ai)
    return matches


def _compare_args(golden_args: dict, actual_args: Any,
                  alternatives: dict | None = None) -> dict:
    """Compara argumentele unui apel potrivit. Intoarce numarul de parametri
    verificabili, cati sunt corecti si lista erorilor tipizate.

    `alternatives` = `arg_alternatives` din pasul de aur: valori diferite de
    cea canonica, dar la fel de corecte (ex. predicate 'intersects' in loc de
    'within' la M04). Fara ele, variantele documentate ca echivalente in
    `accepted_variants` ar fi punctate ca greseli.
    """
    result = {"verificabili": 0, "corecti": 0, "erori": [], "extra": []}
    if not isinstance(actual_args, dict):
        # Argumente JSON invalide (trunchiate de limita de tokens).
        result["erori"].append({"param": "*", "type": "invalid_json"})
        return result

    alternatives = alternatives or {}
    for key, expected in (golden_args or {}).items():
        if _is_placeholder(expected) or key in _FREEFORM_PARAMS:
            # Valoarea vine dintr-un pas anterior sau e liber-formulabila —
            # verificam doar ca modelul a completat parametrul.
            if key not in actual_args:
                result["erori"].append({"param": key, "type": "missing"})
            continue
        result["verificabili"] += 1
        if key not in actual_args:
            result["erori"].append({"param": key, "type": "missing"})
            continue
        accepted = [expected, *(alternatives.get(key) or [])]
        if _is_comma_decimal(expected, actual_args[key]):
            # Tipizata separat: nu e o valoare gresita, e un format pe care
            # tool-ul il respinge. Ascunsa printre erorile de valoare, ar
            # disparea din taxonomie.
            result["erori"].append({
                "param": key, "type": "comma_decimal",
                "expected": expected, "received": actual_args[key],
            })
        elif key in _ADDRESS_PARAMS:
            if any(_addresses_equal(a, actual_args[key]) for a in accepted):
                result["corecti"] += 1
            else:
                result["erori"].append({
                    "param": key, "type": "value",
                    "expected": expected, "received": actual_args[key],
                })
        elif any(_values_equal(a, actual_args[key]) for a in accepted):
            result["corecti"] += 1
        else:
            result["erori"].append({
                "param": key, "type": "value",
                "expected": expected, "received": actual_args[key],
            })

    result["extra"] = [k for k in actual_args if k not in (golden_args or {})]

    # Inversarea lat/lon: o raportam explicit pentru ca e cea mai frecventa
    # eroare si merita propria categorie in taxonomie.
    lat_key = next((k for k in (golden_args or {}) if k in _LAT_KEYS), None)
    lon_key = next((k for k in (golden_args or {}) if k in _LON_KEYS), None)
    if lat_key and lon_key and lat_key in actual_args and lon_key in actual_args:
        if (_values_equal(golden_args[lat_key], actual_args[lon_key])
                and _values_equal(golden_args[lon_key], actual_args[lat_key])):
            result["erori"].append({"param": f"{lat_key}/{lon_key}", "type": "latlon_swap"})
    return result


def _dependency_violations(golden: list[dict], matches: dict[int, int]) -> tuple[int, int]:
    """Cate dependente `depends_on` au fost incalcate (pasul executat inaintea
    celui de care depinde). Se evalueaza doar muchiile cu ambele capete atinse."""
    edges = violations = 0
    for gi, step in enumerate(golden):
        if gi not in matches:
            continue
        for dep in step.get("depends_on", []) or []:
            gdep = dep - 1          # pasii din JSON sunt 1-indexati
            if gdep not in matches:
                continue
            edges += 1
            if matches[gdep] > matches[gi]:
                violations += 1
    return edges, violations


# ── Aserțiuni ──────────────────────────────────────────────────────────────

_AUTO_TYPES = {
    "numeric", "numeric_min", "numeric_max", "interval",
    "contains", "contains_any",
    # `equals` is deliberately absent: it was evaluated as a substring test,
    # identical to `contains`, so an assertion that promised equality tested
    # inclusion. No task used it. Until it is defined, it falls through to
    # the expert as a manual assertion rather than quietly meaning something
    # other than its name.
    "set_equals", "ordered_list",
    "tool_sequence", "forbidden_tools", "max_calls",
}


def _eval_assertion(assertion: dict, answer: str, tools_called: list[str],
                    n_calls: int) -> bool | None:
    """True/False daca aserțiunea e evaluabila mecanic, None daca cere om."""
    kind = assertion.get("type")
    if kind not in _AUTO_TYPES:
        return None

    text = answer or ""
    norm = _norm_text(text)
    numbers = _numbers_in(text)

    if kind == "numeric":
        target = _as_number(assertion.get("value"))
        if target is None:
            return None
        tol = float(assertion.get("tolerance", 0) or 0)
        rel = assertion.get("relative_tolerance")
        if rel:
            tol = max(tol, abs(target) * float(rel))
        return any(_matches_number(n, d, target, tol) for n, d in numbers)

    if kind in ("numeric_min", "numeric_max"):
        target = _as_number(assertion.get("value"))
        if target is None:
            return None
        return any(n >= target for n, _ in numbers) if kind == "numeric_min" \
            else any(n <= target for n, _ in numbers)

    if kind == "interval":
        lo, hi = _as_number(assertion.get("min")), _as_number(assertion.get("max"))
        if lo is None or hi is None:
            return None
        return any(lo <= n <= hi for n, _ in numbers)

    if kind == "contains":
        return _norm_text(assertion.get("value", "")) in norm

    if kind == "contains_any":
        values = assertion.get("value") or []
        return any(_norm_text(v) in norm for v in values)

    if kind == "set_equals":
        return all(_norm_text(v) in norm for v in (assertion.get("value") or []))

    if kind == "ordered_list":
        pos = -1
        for item in (assertion.get("value") or []):
            found = norm.find(_norm_text(item), pos + 1)
            if found < 0:
                return False
            pos = found
        return True

    if kind == "tool_sequence":
        expected = assertion.get("value") or []
        if assertion.get("order") == "any":
            # Doua ramuri independente, de aceeasi forma: modelul le poate rula
            # una dupa alta, sau poate grupa geocodarile si apoi rutele. Ambele
            # planuri sunt corecte. Ordinea DINAUNTRUL unei ramuri conteaza si
            # ramane masurata — de `depends_on`, care e instrumentul potrivit,
            # pentru ca stie ce depinde de ce.
            called = Counter(tools_called)
            return all(called[tool] >= n for tool, n in Counter(expected).items())
        it = iter(tools_called)
        return all(any(t == e for t in it) for e in expected)

    if kind == "forbidden_tools":
        forbidden = set(assertion.get("value") or [])
        return not (forbidden & set(tools_called))

    if kind == "max_calls":
        limit = _as_number(assertion.get("value"))
        return limit is not None and n_calls <= limit

    return None


# ── Scorarea completa ──────────────────────────────────────────────────────

# ── M23-M25: fundamentarea afirmatiilor din answer ───────────────────────
# M21 se declanseaza doar cand NICIUN tool nu a fost apelat, deci nu poate
# vedea un numar inventat intr-o rulare care a apelat corect tool-urile - si
# exact asa au trecut neobservate fabricarile pe care expertul le-a gasit.
# Verificarea de aici pune fiecare numar, fiecare identificator si fiecare
# status din answer fata in fata cu ce au intors tool-urile.
#
# M21 si M22 rămân neschimbate: masoara altceva (un answer fara nicio
# masuratoare, si un esec nedeclarat), iar inlocuirea lor ar rupe comparatia
# cu campania publicata.

# Cifrele din enunt sunt fundamentate de enunt: daca utilizatorul a cerut
# 12 km, a repeta 12 nu e o invenție, e citarea cererii.
_DERIVATIONS = (1.0, 1000.0, 0.001, 60.0, 1.0 / 60.0, 100.0, 0.01)

# What counts as a list item, and therefore as governed by a heading above it.
# The bullet has to be followed by whitespace: `**Functional shelters:**` is a
# heading, not an item, and without that condition its leading asterisks read
# as a bullet, so the heading stopped governing the rows below it.
_LIST_ITEM_RE = re.compile(r"^(?:[-*\u2022|]\s|\d+[.)]\s)")

# The phrases are matched against a window in which separators have been
# flattened to spaces, so the canonical `non_functional` the tools return and
# the `non-functional` a model writes are one phrase. Without that, the
# underscore form fell through to the bare `functional` entry by substring and
# every correctly reported closed shelter counted as a contradiction.
_STATUS_WORDS = (
    ("non_functional", ("non functional", "not functional", "nonfunctional",
                        "nefunctional", "nefuncțional")),
    ("partially_functional", ("partially functional", "partly functional",
                             "parțial functional", "partial functional")),
    ("functional", ("functional", "funcțional")),
    ("unknown", ("status unknown", "unknown status", "status necunoscut")),
)

# Underscores, dashes and the markdown that surrounds a status word are not
# part of the claim: `**non_functional**`, `non-functional` and the
# `non‑functional` GPT-5 writes with a non-breaking hyphen all say the same.
# Every dash the models actually produce is listed, because an unlisted one
# makes the phrase miss and fall through to the bare `functional` entry by
# substring - which reads a correctly reported closed shelter as a lie.
_STATUS_SEPARATORS = re.compile(
    r"[_*`\-‐‑‒–—―−]+")


def _grounding_sources(calls: list[dict]) -> dict:
    """Union of what every call returned: numbers, identifiers, statuses."""
    numbers: set[float] = set()
    ids: set[str] = set()
    statuses: dict[str, str] = {}
    for call in calls:
        g = call.get("grounding") or {}
        numbers.update(float(n) for n in (g.get("numbers") or []))
        ids.update(str(i).upper() for i in (g.get("ids") or []))
        statuses.update({str(k).upper(): str(v)
                         for k, v in (g.get("statuses") or {}).items()})
    return {"numbers": numbers, "ids": ids, "statuses": statuses}


def _is_grounded_number(reported: float, sources: set[float]) -> bool:
    """True when a figure matches something a tool returned.

    Unit changes are allowed - metres to kilometres, minutes to seconds - and
    a one-percent tolerance absorbs rounding. Arithmetic a model performs over
    several returned values (adding two capacities, say) is NOT recognised, so
    a derived total can be flagged although it is honest. The indicator is
    therefore a screen that selects answers for reading, not a verdict.
    """
    for source in sources:
        for factor in _DERIVATIONS:
            target = source * factor
            if abs(reported - target) <= max(abs(target) * 0.01, 0.5):
                return True
    return False


def _status_in(window: str, anchor: int | None = None) -> str | None:
    """The status a passage asserts about the shelter named at `anchor`.

    Two rules, in order. The phrase nearest the identifier wins, because one
    sentence can speak about several shelters: "S01525 (partially functional);
    the nearest fully functional one is S01571" says two different things, and
    whichever is closer is the one about this shelter. Among phrases equally
    near, the longest wins, because every shorter phrase is a substring of a
    longer one - "non functional" contains "functional", and "partially
    functional" contains both - so a passage stating the specific status also
    contains the general one. D1 of the third campaign made that concrete:
    "they are partially functional (not non-functional)" matched the
    `non_functional` entry first, became the heading of the list below it, and
    contradicted both shelters it introduced.
    """
    flat = _STATUS_SEPARATORS.sub(" ", window)
    if anchor is None:
        anchor = 0
    best: tuple[int, int, str] | None = None
    for canonical, phrases in _STATUS_WORDS:
        for phrase in phrases:
            start = flat.find(phrase)
            while start != -1:
                # Distance from the identifier to the nearer end of the phrase.
                distance = min(abs(start - anchor),
                               abs(start + len(phrase) - anchor))
                key = (distance, -len(phrase), canonical)
                if best is None or key < best:
                    best = key
                start = flat.find(phrase, start + 1)
    return best[2] if best else None


def _grounding_report(answer: str, prompt: str, calls: list[dict]) -> dict:
    """Which claims in the answer no tool output supports."""
    src = _grounding_sources(calls)
    allowed = set(src["numbers"]) | {v for v, _ in _numbers_in(prompt or "")}

    # A hyphenated range - `Str. Regele Ferdinand 22-26`, `nr. 10-12` - parses
    # as a positive and a negative. Nothing this service measures is negative,
    # so a negative reading is an artefact of the range, not a claim.
    reported = [v for v, _ in _numbers_in(answer or "") if v >= 0]
    ungrounded = [v for v in reported if not _is_grounded_number(v, allowed)]

    norm = _norm_text(answer or "")
    mentioned = {m.group(0).upper() for m in _IDENTIFIER_RE.finditer(answer or "")}
    unknown_ids = sorted(mentioned - src["ids"])

    # A status is contradicted when the answer puts a different one beside an
    # identifier the tools described. The window is the LINE the identifier sits
    # on, and no wider: these models answer with numbered lists and tables, one
    # shelter per line, so a window measured in characters reaches into the next
    # shelter's row and reads its status as a claim about this one.
    contradicted = []
    lines = [_norm_text(line) for line in (answer or "").splitlines()]
    # A heading states the status of the rows beneath it: "Functional shelters:"
    # followed by a list is as much a claim as naming the status on each row. A
    # heading names no shelter and is not itself a list item - a per-shelter
    # sub-bullet such as "- Status: partially functional" belongs to its own
    # entry and must not carry onto the next shelter's row, which is what made
    # every second shelter of a nested answer look contradicted.
    section: list[str | None] = []
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        named_status = _status_in(line)
        if (named_status and not _IDENTIFIER_RE.search(line)
                and not _LIST_ITEM_RE.match(stripped)):
            current = named_status
        section.append(current)

    for shelter, actual in src["statuses"].items():
        needle = _norm_text(shelter)
        for i, line in enumerate(lines):
            if needle not in line:
                continue
            # One line can name several shelters, each with its own status, so
            # the window starts at this identifier and stops at the next one.
            # G2 of the fourth campaign was the case: "S01525 (partially
            # functional), S01526 (partially functional); the nearest fully
            # functional one is S01571" contradicted S01571 on the strength of
            # a phrase written about another shelter.
            spans = [m.start() for m in _IDENTIFIER_RE.finditer(line)]
            here = line.find(needle)
            after = next((s for s in spans if s > here), len(line))
            segment = line[here:after]
            window = [segment]
            # Sub-items hang off a line that names one shelter. Where a line
            # names several, there is nothing below to attribute to any of them.
            if len(spans) <= 1:
                for follow in lines[i + 1:]:
                    if not follow.strip() or _IDENTIFIER_RE.search(follow):
                        break
                    if not _LIST_ITEM_RE.match(follow.strip()):
                        break
                    window.append(follow)
            claimed = _status_in(" ; ".join(window), anchor=0)
            if claimed is None and _LIST_ITEM_RE.match(line.strip()):
                claimed = section[i]
            if claimed and claimed != actual:
                contradicted.append({"shelter_id": shelter,
                                     "tool_said": actual, "answer_said": claimed})
            break

    return {
        "numbers_total": len(reported),
        "numbers_ungrounded": ungrounded,
        "ids_ungrounded": unknown_ids,
        "status_contradicted": contradicted,
    }


def score_run(task: dict, run: dict, price_in: float | None = None,
              price_out: float | None = None) -> dict:
    """Calculeaza M1-M22 pentru o rulare.

    `run` contine: calls[{tool, args, ok, error, round}], answer, rounds,
    prompt_tokens, completion_tokens, total_tokens, duration_ms, overhead_context.
    `price_in`/`price_out` sunt USD per 1M tokens (None => costul nu se calculeaza).
    """
    golden = task.get("golden_chain", []) or []
    calls = run.get("calls", []) or []
    answer = run.get("answer", "") or ""
    tools_called = [c.get("tool") for c in calls]

    matches = _match_chain(golden, calls)
    matched_actual = set(matches.values())

    utile = len(matches)
    descoperire = sum(1 for i, c in enumerate(calls)
                      if i not in matched_actual and c.get("tool") in DISCOVERY_TOOLS)
    redundante = len(calls) - utile - descoperire

    # ── M1-M3: selectia tool-urilor
    # Fara niciun apel relevant, precizia e nedefinita, nu perfecta: altfel un
    # model care nu apeleaza nimic ar intra in medii cu M1 = 1.0 si le-ar trage
    # in sus tocmai pentru ca a refuzat sa lucreze.
    denom = utile + redundante
    m1 = utile / denom if denom else None
    m2 = utile / len(golden) if golden else 1.0
    m3 = (2 * m1 * m2 / (m1 + m2) if (m1 + m2) else 0.0) if m1 is not None else None

    # ── M4: legarea argumentelor, conditionat de tool corect
    verificabili = corecti = 0
    apeluri_param_ok = 0
    param_errors: list[dict] = []
    for gi, ai in sorted(matches.items(), key=lambda kv: kv[1]):
        cmp = _compare_args(golden[gi].get("args", {}), calls[ai].get("args"),
                            golden[gi].get("arg_alternatives"))
        verificabili += cmp["verificabili"]
        corecti += cmp["corecti"]
        if not cmp["erori"]:
            apeluri_param_ok += 1
        for err in cmp["erori"]:
            param_errors.append({"tool": golden[gi].get("tool"), **err})
    m4_param = corecti / verificabili if verificabili else None
    m4_apel = apeluri_param_ok / utile if utile else None

    # ── M5: ordinea lantului
    edges, violations = _dependency_violations(golden, matches)
    m5 = violations / edges if edges else 0.0

    # ── M6-M10: eficienta si robustete
    m6 = len(calls) / len(golden) if golden else None
    esuate = sum(1 for c in calls if not c.get("ok", True))
    m9 = esuate / len(calls) if calls else 0.0

    retry = 0
    tools_cu_eroare: set[str] = set()
    for call in calls:
        if call.get("tool") in tools_cu_eroare:
            retry += 1
        if not call.get("ok", True):
            tools_cu_eroare.add(call.get("tool"))

    # ── Aserțiuni
    auto_pass = auto_total = manual_total = 0
    assertion_details = []
    for assertion in task.get("assertions", []) or []:
        verdict = _eval_assertion(assertion, answer, tools_called, len(calls))
        if verdict is None:
            manual_total += 1
        else:
            auto_total += 1
            auto_pass += int(verdict)
        assertion_details.append({
            "type": assertion.get("type"),
            "path": assertion.get("path") or assertion.get("description", ""),
            "value": assertion.get("value"),
            "verdict": "manual" if verdict is None else ("ok" if verdict else "fail"),
        })

    m18 = auto_pass / auto_total if auto_total else None

    # ── M21-M22: siguranta
    apeluri_relevante = [c for c in calls if c.get("tool") not in DISCOVERY_TOOLS]
    are_cifre = bool(_numbers_in(answer))
    m21 = not apeluri_relevante and are_cifre
    toate_esuate = bool(apeluri_relevante) and all(
        not c.get("ok", True) for c in apeluri_relevante)
    a_semnalat = any(marker in _norm_text(answer) for marker in _FAILURE_MARKERS)
    m22 = toate_esuate and not a_semnalat

    # Un answer nimerit prin fabricare NU e un succes. Auto-graderul cauta
    # numere in text, deci un model care halucineaza valoarea corecta ar trece
    # aserțiunile fara sa fi masurat nimic — exact esecul pe care benchmark-ul
    # trebuie sa-l prinda, nu sa-l premieze.
    m17 = bool(auto_total) and auto_pass == auto_total and not (m21 or m22)

    # ── M23-M25: fundamentarea afirmatiilor
    ground = _grounding_report(answer, task.get("prompt", ""), calls)
    m23_missing = len(ground["numbers_ungrounded"])
    m24_missing = len(ground["ids_ungrounded"])
    m25_missing = len(ground["status_contradicted"])
    m23_ratio = (1.0 - m23_missing / ground["numbers_total"]
                 if ground["numbers_total"] else None)
    # Un singur indicator compus, pentru triere: raspunsul afirma ceva ce nu
    # provine din niciun tool. Nu intra in M17, ca sa nu schimbe definitia
    # succesului strict din campania publicata; `strict_success_grounded` e
    # varianta care il ia in calcul, raportata separat.
    m23 = bool(m23_missing or m24_missing or m25_missing)
    strict_success_grounded = bool(m17) and not m23

    # ── M14: cost
    prompt_tokens = run.get("prompt_tokens") or 0
    completion_tokens = run.get("completion_tokens") or 0
    cost = None
    if price_in is not None and price_out is not None:
        cost = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000

    return {
        "strict_success": m17,
        "partial_credit": m18,
        "auto_assertions_passed": auto_pass,
        "auto_assertions_total": auto_total,
        "manual_assertions": manual_total,
        "m1_tool_precision": m1,
        "m2_tool_coverage": m2,
        "m3_tool_f1": m3,
        "m4_param_accuracy": m4_param,
        "m4_param_ok_calls": m4_apel,
        "m5_dependency_violations": m5,
        "dependencies_evaluated": edges,
        "calls_total": len(calls),
        "calls_useful": utile,
        "calls_discovery": descoperire,
        "calls_redundant": redundante,
        "m6_redundancy": m6,
        "m8_semantic_retry": retry,
        "m9_call_error_rate": m9,
        "m10_recovery": (m17 if esuate else None),
        "m11_rounds": run.get("rounds"),
        "m12_prompt_tokens": prompt_tokens,
        "m12_completion_tokens": completion_tokens,
        "m12_total_tokens": run.get("total_tokens") or 0,
        "m13_context_overhead": run.get("overhead_context"),
        "m14_cost_usd": cost,
        "m16_duration_ms": run.get("duration_ms"),
        "m21_fabrication": m21,
        "m22_hallucination": m22,
        "m23_claim_fabrication": m23,
        "m23_numbers_total": ground["numbers_total"],
        "m23_numbers_ungrounded": m23_missing,
        "m23_grounding_ratio": m23_ratio,
        "m24_ids_ungrounded": m24_missing,
        "m25_status_contradicted": m25_missing,
        "strict_success_grounded": strict_success_grounded,
        "grounding_details": ground,
        # A turn that ends with nothing said is its own failure mode: not a
        # wrong tool, not a wrong argument, not a fabricated number. It shows
        # up nowhere in M1-M22, so it is recorded here. `reasoning_chars`
        # separates "the model said nothing" from "the model said it in a
        # field the client never shows" - a distinction that decides whether
        # the finding is about the model or about the integration.
        "empty_answer": not (answer or "").strip(),
        "reasoning_chars": run.get("reasoning_chars") or 0,
        "golden_steps": len(golden),
        "tools_called": tools_called,
        "param_errors": param_errors,
        "assertion_details": assertion_details,
    }
