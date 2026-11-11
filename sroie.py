"""SROIE loader + field normalizers for the scaled study (see OBJECTIVE.md).

Scoring is key-by-key on the GT fields (company, date, address, total). Value
normalization lives HERE, in the scorer — schemas stay loose on value format, otherwise
we'd measure format compliance instead of reading (GT itself mixes `25/12/2018`,
`12-01-19`, `05 MAR 2018`, `$8.20`, `1,007.50`...).

`uv run sroie.py` runs the self-checks and prints a GT-vs-normalized smoke test.
"""
import re
from datetime import date

from datasets import load_dataset
from rapidfuzz.distance import Levenshtein

REPO = "jsdnrs/ICDAR2019-SROIE"
FIELDS = ("company", "date", "address", "total")
COLUMNS = FIELDS + ("date_lenient",)  # strict fields + the lenient date, as a second column
# company/address: GT carries OCR noise ("BOOK TA .K"), so exact match is too harsh.
# Normalized Levenshtein similarity (1 - edits/max_len), not difflib: difflib's autojunk
# + longest-block recursion scored 3 typos in a 244-char address at 0.42.
FUZZY = 0.9
MONTHS = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}


def load_split(split="test"):
    return load_dataset(REPO, split=split)


def iter_samples(split="test"):
    """Yield (key, PIL image, GT dict) one receipt at a time; never materialize the images
    (a list of ~1000 decoded receipts OOM-killed the machine once)."""
    for row in load_split(split):
        image = row["image"]
        if image.mode != "RGB":  # some scans are grayscale: the LFM processor rejects 1-channel images
            src_format = image.format
            image = image.convert("RGB")
            image.format = src_format  # convert() drops it; outlines' Image() needs it
        yield row["key"], image, row["entities"]


def norm_date(s):
    """Any GT/model date -> ISO 'YYYY-MM-DD', or None if unparseable.
    shortcut: numeric dates are read day-first (Malaysian receipts); a month-first GT is
    only recovered when unambiguous (day>12, e.g. 12/28/2017) — a US 03/04/2018 stays
    misparsed. Fine for SROIE (3 such rows, all unambiguous), revisit for another dataset."""
    t = re.findall(r"[A-Za-z]+|\d+", s or "")
    nums = [x for x in t if x.isdigit()]
    alpha = [x for x in t if x.isalpha()]
    try:
        if len(t) == 1 and len(t[0]) == 8:  # 20180304 or 25032018
            a = t[0]
            y, m, d = (a[:4], a[4:6], a[6:]) if a[:2] in ("19", "20") else (a[4:], a[2:4], a[:2])
        elif alpha:  # 05 MAR 2018 | OCT 3, 2016 | 05 MAR 18
            m = MONTHS[alpha[0][:3].upper()]
            d, y = (nums[1], nums[0]) if len(nums[0]) == 4 else (nums[0], nums[1])
        elif len(nums) == 3:
            y, m, d = nums if len(nums[0]) == 4 else (nums[2], nums[1], nums[0])
            if int(m) > 12 >= int(d):  # month-first GT with an unambiguous day (12/28/2017)
                m, d = d, m
        else:
            return None
        y = int(y) + 2000 if len(str(y)) == 2 else int(y)
        return date(y, int(m), int(d)).isoformat()
    except (ValueError, KeyError, IndexError):
        return None


def norm_total(s):
    """'$8.20' / 'RM 33.90' / '1,007.50' / '27,49' / -1.73 -> float, or None if empty."""
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", "" if s is None else str(s))
    if not m:
        return None
    x = m.group()
    if "," in x and "." not in x and re.fullmatch(r"-?\d+,\d{1,2}", x):
        x = x.replace(",", ".")  # French decimal comma
    return float(x.replace(",", ""))


def norm_text(s):
    return re.sub(r"[\W_]", "", ("" if s is None else str(s)).casefold())


def field_ok(field, pred, gt):
    """True/False, or None when the GT is missing/unparseable (drop from that field's denominator)."""
    if field == "total":
        g, p = norm_total(gt), norm_total(pred)
        return None if g is None else (p is not None and abs(p - g) < 0.01)
    if field == "date":
        g = norm_date(gt)
        return None if g is None else norm_date(pred) == g
    g, p = norm_text(gt), norm_text(pred)
    return None if not g else Levenshtein.normalized_similarity(p, g) >= FUZZY


# Date candidates inside a longer string: 15/01/2019, 05 MAR 2018, OCT 3, 2016, 20180304
DATE_RE = re.compile(
    r"\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}|\d{1,2}[ /.\-]?[A-Za-z]{3,9}[ /.\-]?\d{2,4}"
    r"|[A-Za-z]{3,9} \d{1,2}, ?\d{4}|\b\d{8}\b"
)


def date_lenient(pred, gt):
    """Lenient date: the GT date appears ANYWHERE in the prediction ('15/01/2019 11:05:16 AM'
    passes). Reported next to the strict `date`, which needs the field to hold only the date."""
    g = norm_date(gt)
    if g is None:
        return None
    p = "" if pred is None else str(pred)
    return norm_date(p) == g or any(norm_date(m.group()) == g for m in DATE_RE.finditer(p))


def score_fields(pred, gt):
    """{column: True/False/None} for every COLUMNS entry; pred = dict of raw model values."""
    ok = {f: field_ok(f, pred.get(f), gt[f]) for f in FIELDS}
    ok["date_lenient"] = date_lenient(pred.get("date"), gt["date"])
    return ok


if __name__ == "__main__":
    # --- self-checks: every odd format found in the GT survey ---
    for raw, iso in {
        "25/12/2018": "2018-12-25", "12-01-19": "2019-01-12", "05 MAR 2018": "2018-03-05",
        "05 MAR 18": "2018-03-05", "03/MAR/2018": "2018-03-03", "2018-12-25": "2018-12-25",
        "20180304": "2018-03-04", "25032018": "2018-03-25", "11.02.18": "2018-02-11",
        "(06/12/2016)": "2016-12-06", "OCT 3, 2016": "2016-10-03", "1/2/2019": "2019-02-01",
        "12/28/2017": "2017-12-28", "4/22/2018": "2018-04-22",  # month-first, day>12
    }.items():
        assert norm_date(raw) == iso, (raw, norm_date(raw), iso)
    assert norm_date("32/13/2018") is None and norm_date("") is None
    for raw, val in {"$8.20": 8.2, "RM 33.90": 33.9, "1,007.50": 1007.5, "-1.73": -1.73,
                     "27,49 EUR": 27.49, 27.49: 27.49, "": None, None: None}.items():
        assert norm_total(raw) == val, (raw, norm_total(raw), val)
    assert field_ok("total", "RM33.90", "33.90") and not field_ok("total", "33.92", "33.90")
    assert field_ok("total", 8.2, "") is None  # empty GT -> excluded, not counted wrong
    assert field_ok("company", "Book Ta .K (Taman Daya) Sdn Bhd", "BOOK TA .K (TAMAN DAYA) SDN BHD")
    assert not field_ok("company", "INDAH GIFT & HOME DECO", "MR D.I.Y. (JOHOR) SDN BHD")
    long_addr = "LOT 1851-A & 1851-B, JALAN KPB 6, KAWASAN PERINDUSTRIAN BALAKONG, SERI KEMBANGAN " * 3
    assert field_ok("address", long_addr[:100] + "XYZ" + long_addr[103:], long_addr)  # 3 typos, 250 chars
    both = "15/01/2019 11:05:16 AM"  # date + time stuffed in the date field
    assert date_lenient(both, "15/01/2019") and not field_ok("date", both, "15/01/2019")
    assert date_lenient("le 05 MAR 2018 à 10:00", "05 MAR 2018")
    assert not date_lenient("13/06/2023 13:06:09", "05/01/2026") and date_lenient("x", "") is None
    assert set(score_fields({}, dict(company="a", date="01/01/2019", address="b", total="1.0"))) == set(COLUMNS)
    print("self-checks OK")

    # --- GT-only smoke test (select_columns => images never decoded) ---
    for split in ("train", "test"):
        ds = load_split(split).select_columns(["key", "entities"])
        bad = {f: sum(field_ok(f, "", e[f]) is None for e in ds["entities"]) for f in ("date", "total")}
        print(f"\n{split}: {len(ds)} receipts; GT missing/unparseable (excluded from denominators): {bad}")
        if split == "test":
            for key, e in zip(ds["key"][:5], ds["entities"][:5]):
                print(f"  {key}  date {e['date']!r} -> {norm_date(e['date'])}   "
                      f"total {e['total']!r} -> {norm_total(e['total'])}")
