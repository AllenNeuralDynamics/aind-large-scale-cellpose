"""Write an aind-data-schema (>=2.9) processing.json for a Code Ocean capsule step and
append it to the dataset's top-level /results/processing.json.

This version assumes LINEAR pipelines: each capsule inherits a single upstream history and
chains its process(es) onto the end. Parallel worker steps are expected to collapse into a
single processing.json entry (one capsule writes it, not each worker). If a step ever has to
combine two *different* upstream histories (a true fan-in / collector), this helper raises
rather than guess -- use the DAG-merging version for that.

Usage in a capsule's run script:

    from processing_helper import (ProcessTimer, make_data_process,
                                   input_assets_from_data, write_processing)

    with ProcessTimer() as t:
        results = run_step(params)

    process = make_data_process(
        process_type="Spike sorting",          # a ProcessName value
        name="Spike sorting ProbeA",           # unique & descriptive if the type can repeat
        code_url="https://github.com/AllenNeuralDynamics/my-capsule",
        code_version="1.2.0",                  # and/or commit_hash=git_commit()
        experimenters=["Jane Doe"],
        start=t.start, end=t.end,
        parameters=effective_params,           # ALL effective params, incl. defaults
        input_data=input_assets_from_data("/data"),
        output_path="spikesorted",             # relative to /results
        core_dependency=("spikeinterface", None),   # version looked up when None
    )
    write_processing([process])                # inherits upstream in /data, appends, writes /results
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import platform
import shutil
import subprocess
import warnings
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from aind_data_schema.components.identifiers import Code, DataAsset, Software
from aind_data_schema.core.processing import DataProcess, Processing, ProcessStage, ResourceUsage
from aind_data_schema_models.process_names import ProcessName
from aind_data_schema_models.units import MemoryUnit

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- small utilities
def now_utc() -> dt.datetime:
    """Timezone-aware 'now' (the schema rejects naive datetimes)."""
    return dt.datetime.now(dt.timezone.utc)


class ProcessTimer:
    """Context manager recording tz-aware start/end around the actual work of a step."""

    def __enter__(self) -> "ProcessTimer":
        self.start = now_utc()
        self.end: Optional[dt.datetime] = None
        return self

    def __exit__(self, *exc) -> None:
        self.end = now_utc()


def to_jsonable(obj: Any) -> Any:
    """Make numpy / Path / Enum / datetime / pydantic values JSON-safe.

    Kept because `parameters` is usually a grab-bag of effective args (numpy scalars,
    argparse.Namespace, Paths) that pydantic will not serialize on its own.
    """
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump(mode="json"))
    if hasattr(obj, "__dict__") and not isinstance(obj, type):  # argparse.Namespace, dataclasses
        return to_jsonable(vars(obj))
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def package_version(name: str) -> Optional[str]:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def git_commit(repo_dir: Union[str, Path] = "/code") -> Optional[str]:
    """Commit hash of the capsule code, if the run has a git checkout (not always true)."""
    if not shutil.which("git"):
        return None
    try:
        out = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


# ---- optional resource provenance: delete this block + the `resources=` call if unused -----
def collect_resources() -> ResourceUsage:
    """OS / arch / core count / memory limit of the machine the step ran on."""
    mem = _memory_gb()
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count()
    return ResourceUsage(
        os=f"{platform.system()} {platform.release()}",
        architecture=platform.machine(),
        cpu_cores=cores,
        system_memory=mem,
        system_memory_unit=MemoryUnit.GB if mem else None,
    )


def _memory_gb() -> Optional[float]:
    try:  # cgroup v2 container limit
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if raw.isdigit() and int(raw) < 1 << 60:
            return round(int(raw) / 1024**3, 2)
    except OSError:
        pass
    try:  # host memory
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2)
    except (ValueError, OSError, AttributeError):
        return None
# --------------------------------------------------------------------------------------------


def input_assets_from_data(data_dir: Union[str, Path] = "/data") -> List[DataAsset]:
    """One DataAsset per attached asset, named from its data_description.json when present."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    assets = []
    for child in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        name = child.name
        dd = child / "data_description.json"
        if dd.exists():
            try:
                name = json.loads(dd.read_text()).get("name") or name
            except Exception:  # noqa: BLE001
                pass
        assets.append(DataAsset(name=name))
    return assets


def _as_assets(input_data) -> Optional[List[DataAsset]]:
    if not input_data:
        return None
    out = []
    for item in input_data:
        if isinstance(item, DataAsset):
            out.append(item)
        elif isinstance(item, dict):
            out.append(DataAsset(**item))
        elif str(item).startswith(("s3://", "http://", "https://")):
            out.append(DataAsset(url=str(item)))
        else:
            out.append(DataAsset(name=str(item)))
    return out


# --------------------------------------------------------------------------- authoring one step
def make_data_process(
    process_type: Union[str, ProcessName],
    code_url: str,
    experimenters: Sequence[str],
    start: dt.datetime,
    end: Optional[dt.datetime] = None,
    code_version: Optional[str] = None,
    commit_hash: Optional[str] = None,
    name: Optional[str] = None,
    stage: Union[str, ProcessStage] = ProcessStage.PROCESSING,
    parameters: Optional[Any] = None,
    input_data: Optional[Iterable[Union[str, DataAsset, dict]]] = None,
    output_path: Optional[str] = None,
    output_parameters: Optional[Any] = None,
    notes: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    core_dependency: Optional[Tuple[str, Optional[str]]] = None,
    resources: Optional[ResourceUsage] = None,
    run_script: Optional[str] = "code/run",
    language: Optional[str] = "Python",
    language_version: Optional[str] = None,
) -> DataProcess:
    """Build one DataProcess; raises on inputs that would make the record wrong or unusable."""
    if not code_version and not commit_hash:
        raise ValueError("Provide code_version and/or commit_hash.")
    if not experimenters:
        raise ValueError("experimenters must contain at least one name.")
    end = end or now_utc()
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start}).")
    if output_path and Path(output_path).is_absolute():
        raise ValueError("output_path must be relative to /results (e.g. 'spikesorted').")
    if str(process_type) in (ProcessName.OTHER.value, ProcessName.ANALYSIS.value) and not (name and notes):
        raise ValueError("For 'Other'/'Analysis' give both a specific name and notes describing the step.")

    if language == "Python" and language_version is None:
        language_version = platform.python_version()
    dep = None
    if core_dependency:
        dep_name, dep_version = core_dependency
        dep = Software(name=dep_name, version=dep_version or package_version(dep_name))

    return DataProcess(
        process_type=process_type,
        name=name or "",
        stage=stage,
        code=Code(
            url=code_url,
            version=code_version,
            commit_hash=commit_hash,
            run_script=run_script,
            language=language,
            language_version=language_version,
            input_data=_as_assets(input_data),
            parameters=to_jsonable(parameters) if parameters is not None else None,
            core_dependency=dep,
        ),
        experimenters=list(experimenters),
        pipeline_name=pipeline_name,
        start_date_time=start,
        end_date_time=end,
        output_path=output_path,
        output_parameters=to_jsonable(output_parameters) if output_parameters is not None else None,
        notes=notes,
        resources=resources,
    )


# --------------------------------------------------------------------------- loading upstream files
def load_processing_file(path: Union[str, Path]) -> Processing:
    """Load a v2 processing.json. Raises a clear error on a v1 file ('processing_pipeline').

    If you still have v1 data in /data, upgrade the files out-of-band with
    aind-metadata-upgrader rather than carrying an in-process upgrade path in every capsule.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if "processing_pipeline" in data:
        raise ValueError(
            f"{path} is a v1 processing.json. Upgrade it with aind-metadata-upgrader before merging."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Processing.model_validate(data)


def find_upstream_processing(
    data_dir: Union[str, Path] = "/data",
    patterns: Sequence[str] = ("processing.json", "*/processing.json"),
) -> List[Path]:
    """Top-level processing.json of /data and of each attached asset (not deeper, by default)."""
    data_dir = Path(data_dir)
    found: List[Path] = []
    for pat in patterns:
        found.extend(sorted(data_dir.glob(pat)))
    return list(dict.fromkeys(p.resolve() for p in found))


# --------------------------------------------------------------------------- appending (linear)
def _graph_of(proc: Processing) -> Dict[str, List[str]]:
    if proc.dependency_graph:
        return {k: list(v) for k, v in proc.dependency_graph.items()}
    # foreign/old file with no graph: assume its processes ran in sequence
    names = [p.name for p in sorted(proc.data_processes, key=lambda p: p.start_date_time)]
    return {n: ([names[i - 1]] if i else []) for i, n in enumerate(names)}


def _unique_name(base: str, taken: set) -> str:
    if base not in taken:
        return base
    i = 2
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


def _append_linear(
    upstream: Optional[Processing],
    new_processes: Sequence[DataProcess],
    pipelines: Optional[Sequence[Code]] = None,
    notes: Optional[str] = None,
) -> Processing:
    """Carry one upstream history forward unchanged and chain this capsule's processes onto its leaves."""
    processes: List[DataProcess] = []
    graph: Dict[str, set] = {}
    taken: set = set()
    all_notes: List[str] = []
    all_pipelines: List[Code] = []

    if upstream is not None:
        processes.extend(upstream.data_processes)
        taken.update(p.name for p in upstream.data_processes)
        graph = {k: set(v) for k, v in _graph_of(upstream).items()}
        all_pipelines.extend(upstream.pipelines or [])
        if upstream.notes:
            all_notes.extend(n for n in upstream.notes.split("\n") if n)

    has_dependents = {d for deps in graph.values() for d in deps}
    leaves = [n for n in graph if n not in has_dependents]

    prev: Optional[str] = None
    for p in new_processes:
        nm = _unique_name(p.name, taken)   # guard against a name already used upstream
        taken.add(nm)
        if nm != p.name:
            p = p.model_copy(update={"name": nm}, deep=True)
        processes.append(p)
        graph[nm] = set([prev] if prev else leaves)
        prev = nm

    all_pipelines.extend(pipelines or [])
    seen, deduped = set(), []
    for c in all_pipelines:
        k = json.dumps(c.model_dump(mode="json"), sort_keys=True)
        if k not in seen:
            seen.add(k)
            deduped.append(c)
    if notes:
        all_notes.append(notes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Processing(
            data_processes=processes,
            dependency_graph={k: sorted(v) for k, v in graph.items()},
            pipelines=deduped or None,
            notes="\n".join(dict.fromkeys(all_notes)) or None,
        )


# --------------------------------------------------------------------------- writing
def write_processing(
    new_processes: Sequence[DataProcess],
    output_dir: Union[str, Path] = "/results",
    upstream: Union[str, Sequence[Union[str, Path]], None] = "auto",
    pipelines: Optional[Sequence[Code]] = None,
    notes: Optional[str] = None,
    data_dir: Union[str, Path] = "/data",
) -> Path:
    """Inherit the upstream processing.json, append this step's processes, and write /results.

    upstream="auto" finds /data/processing.json and /data/*/processing.json; pass explicit
    paths when inputs mount elsewhere, or None for a step that must not inherit history.

    Raises if more than one *different* upstream history is found: that is a fan-in, which this
    linear helper does not merge. (Identical copies of the same file are collapsed.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = find_upstream_processing(data_dir) if upstream == "auto" else [Path(p) for p in (upstream or [])]
    histories = [load_processing_file(p) for p in paths]

    distinct: Dict[str, Processing] = {h.model_dump_json(): h for h in histories}
    if len(distinct) > 1:
        raise RuntimeError(
            f"Found {len(distinct)} different upstream processing.json histories in {data_dir} "
            f"({[str(p) for p in paths]}). This linear helper inherits one history; a fan-in / "
            "collector step needs the DAG-merging version of this module."
        )
    upstream_proc = next(iter(distinct.values())) if distinct else None
    logger.info("Inheriting %d upstream file(s): %s", len(paths), [str(p) for p in paths])

    merged = _append_linear(upstream_proc, new_processes, pipelines=pipelines, notes=notes)
    merged.write_standard_file(output_directory=output_dir)
    out = output_dir / Processing.default_filename()

    # guard: never silently drop upstream history
    written = Processing.model_validate_json(out.read_text())
    if upstream_proc is not None:
        names = {p.name for p in written.data_processes}
        lost = [p.name for p in upstream_proc.data_processes if p.name not in names]
        if lost:
            raise RuntimeError(f"Merged processing.json dropped upstream processes: {lost}")
    logger.info("Wrote %s: %d processes (%d new)", out, len(written.data_processes), len(new_processes))
    return out