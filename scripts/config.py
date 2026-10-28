"""shared paths, seeds, codes and settings for every pipeline script

change a value here and every step picks it up. run_full_pipeline.py dumps all the
upper-case settings below into outputs/provenance/run_manifest.json on each run
"""
import atexit
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ paths
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
FS_SAV = DATA_DIR / "fs.sav"

# one output folder per analysis stage
OUTPUT_ROOT = ROOT / "outputs"
OUT_DATA_AUDIT = OUTPUT_ROOT / "01_data_audit"
OUT_ANALYTIC = OUTPUT_ROOT / "02_analytic_dataset"
OUT_PROFILES = OUTPUT_ROOT / "03_cluster_profiles"
OUT_OUTCOMES = OUTPUT_ROOT / "04_outcomes_by_cluster"
OUT_PRELIM_SHAP = OUT_OUTCOMES / "shap_preliminary"
OUT_REGRESSION = OUTPUT_ROOT / "05_survey_regression"
OUT_K_COMPARE = OUTPUT_ROOT / "06_k_comparison"
OUT_EXPLAINABLE = OUTPUT_ROOT / "07_explainable_ml"
OUT_PROVENANCE = OUTPUT_ROOT / "provenance"

ANALYTIC_CSV = OUT_ANALYTIC / "analytic.csv"
CLUSTERS_CSV = OUT_PROFILES / "clusters.csv"

# ------------------------------------------------------------------ random seeds
# per-bootstrap-iteration seeds in 03/06 are the loop index b (0..BOOT-1)
SEED = 42                  # kmeans, MCA, bootstrap resampling, cv splits, classifiers
SELF_CHECK_SEED = 0        # 05 svy_cov self-check data
STABILITY_SEED_START = 100  # 06 seed-stability reruns use seeds 100..109

# ------------------------------------------------------------------ sample (codes confirmed against the spss value labels)
ATTENDING, PRIMARY, GRADES, COMPLETED = 1, 1, [2, 3], 1


def stage_a_mask(df):
    """grade 2-3 primary pupils attending school this year"""
    return (df["CB7"] == ATTENDING) & (df["CB8A"] == PRIMARY) & df["CB8B"].isin(GRADES)


# ------------------------------------------------------------------ outcomes
N_WORDS = 72
MIN_WORD_SHARE = 0.9       # reads >= 90% of the story (65 words)
OUTCOMES_ALL = ["reading_strict", "numeracy_strict", "reading_lenient", "numeracy_lenient",
                "reading_strict_incl"]
OUTCOMES_PRIMARY = ["reading_strict", "numeracy_strict"]

# ------------------------------------------------------------------ clustering (03, 06, 07)
FEATURE_PREFIX = "f_"      # clustering features in analytic.csv; d_* are descriptive only
MCA_COMPONENTS = 10
MCA_N_ITER = 10
MCA_KEEP_MIN, MCA_KEEP_MAX = 2, 6
KS_PROFILES = list(range(2, 7))
BOOT_PROFILES = 100
AUTO_K_MIN = 3
AUTO_K_MIN_CLUSTER_SHARE = 0.08
AUTO_K_FALLBACK = 3
KS_COMPARE = list(range(2, 9))
BOOT_COMPARE = 200
DIMS_COMPARE = [3, 4, 6]   # 6 = the dimensionality used in 03

# ------------------------------------------------------------------ explainable ml (07)
CV_REPEATS = 3
CV_FOLDS = 5
DOMAINS = {
    "Socioeconomic": ["wealth", "mother_edu", "area"],
    "Home learning & school contact": ["books", "hw_help", "pta", "school_visit"],
    "Child functioning": ["disability", "emo_difficulty"],
    "Discipline/protection": ["psych_aggr", "phys_punish"],
    "Digital access": ["online_learning", "digital_resources"],
    "Child work": ["econ_work"],
}

# ------------------------------------------------------------------ candidate variables audited in 01
CANDIDATES = {
    "socioeconomic/demographic": ["windex5", "wscore", "melevel", "HH6", "HH7", "HL4", "FSAGE"],
    "home learning/parental involvement": ["PR3", "PR5", "PR6", "PR8", "PR10", "PR11B"],
    "functioning/wellbeing": ["fsdisability", "FCF19", "FCF20", "FCF21", "FCF24", "FCF25", "FCF26"],
    "discipline": ["FCD2C", "FCD2D", "FCD2F", "FCD2G", "FCD2H", "FCD2I", "FCD2J"],
    "digital access": ["CB10D", "CB10EA", "CB10EB", "CB10EC", "CB10ED", "CB10EE", "CB10F"],
    "child labour": ["CL1A", "CL1B", "CL1C", "CL1X", "CL3", "CL4", "CL5", "CL7", "CL9",
                     "CL11A", "CL11B", "CL11C", "CL11D", "CL11E", "CL11F", "CL13"],
    "other": ["CB11", "fsinsurance", "PR12", "PR13"],
}
SPECIAL = ("NO RESPONSE", "DK", "DON'T KNOW", "MISSING", "REFUSED", "NOT STATED")


# ------------------------------------------------------------------ helpers
def ensure(path):
    """create a folder (and parents) if needed and return it"""
    path.mkdir(parents=True, exist_ok=True)
    return path


class _Tee:
    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, s):
        self._stream.write(s)
        if not self._log.closed:  # atexit closes the log before the interpreter's final flush
            self._log.write(s)
        return len(s)

    def flush(self):
        self._stream.flush()
        if not self._log.closed:
            self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def log_console(out_dir, name="console_log.txt"):
    """mirror everything a script prints (stdout and stderr) into out_dir/console_log.txt"""
    log = open(ensure(out_dir) / name, "w", encoding="utf-8", buffering=1)
    atexit.register(log.close)
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)


def _jsonable(v):
    if isinstance(v, Path):
        return str(v.relative_to(ROOT)) if ROOT in v.parents or v == ROOT else str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, range)):
        return [_jsonable(x) for x in v]
    return v


def settings_snapshot():
    """every upper-case setting above, as plain json-able values (used by the run manifest)"""
    return {k: _jsonable(v) for k, v in globals().items() if k.isupper() and not k.startswith("_")}
