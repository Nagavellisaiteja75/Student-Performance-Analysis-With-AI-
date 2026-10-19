"""Download and safely extract the UCI Student Performance archive."""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATASET_FILES, UCI_DATASET_URL


class DatasetDownloadError(RuntimeError):
    """Raised when the UCI archive cannot be downloaded or validated."""


def _expected_members(archive: zipfile.ZipFile) -> dict[str, str]:
    """Map each required filename to the matching archive member, if present."""

    archive_members = archive.namelist()
    return {
        filename: matching_member
        for filename in DATASET_FILES
        if (
            matching_member := next(
                (member for member in archive_members if member.endswith(filename)),
                None,
            )
        )
        is not None
    }


def _extract_expected_files(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract CSVs from direct UCI archives or a nested `student.zip` archive."""

    matching_members = _expected_members(archive)
    if len(matching_members) == len(DATASET_FILES):
        _write_expected_files(archive, matching_members, destination)
        return

    nested_archives = [
        member for member in archive.namelist() if member.lower().endswith(".zip")
    ]
    for nested_member in nested_archives:
        with zipfile.ZipFile(io.BytesIO(archive.read(nested_member))) as nested_archive:
            nested_matches = _expected_members(nested_archive)
            if len(nested_matches) == len(DATASET_FILES):
                _write_expected_files(nested_archive, nested_matches, destination)
                return

    missing_files = sorted(set(DATASET_FILES).difference(matching_members))
    raise DatasetDownloadError(
        "Archive is missing the expected files, including: " + ", ".join(missing_files)
    )


def _write_expected_files(
    archive: zipfile.ZipFile,
    matching_members: dict[str, str],
    destination: Path,
) -> None:
    """Copy only validated dataset CSV members into the local raw-data folder."""

    for filename in DATASET_FILES:
        with archive.open(matching_members[filename]) as source, (destination / filename).open(
            "wb"
        ) as target:
            shutil.copyfileobj(source, target)


def download_uci_dataset(
    destination: Path,
    *,
    force: bool = False,
    timeout_seconds: int = 60,
) -> dict[str, object]:
    """Fetch the UCI archive and extract only the two expected CSV files.

    The function is idempotent by default: it keeps an already-complete local
    download unless ``force`` is requested.
    """

    destination.mkdir(parents=True, exist_ok=True)
    expected_paths = [destination / filename for filename in DATASET_FILES]
    if not force and all(path.exists() for path in expected_paths):
        return {
            "downloaded": False,
            "files": [str(path) for path in expected_paths],
            "message": "Expected raw files already exist; download skipped.",
        }

    archive_path = destination / "student_performance.zip"
    request = urllib.request.Request(
        UCI_DATASET_URL,
        headers={"User-Agent": "student-performance-analytics/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with archive_path.open("wb") as archive_file:
                shutil.copyfileobj(response, archive_file)

        with zipfile.ZipFile(archive_path) as archive:
            _extract_expected_files(archive, destination)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile) as error:
        raise DatasetDownloadError(
            "Could not download and extract the UCI Student Performance dataset. "
            "Check your internet connection and retry."
        ) from error
    finally:
        archive_path.unlink(missing_ok=True)

    return {
        "downloaded": True,
        "files": [str(path) for path in expected_paths],
        "message": "UCI raw files downloaded and validated successfully.",
    }
