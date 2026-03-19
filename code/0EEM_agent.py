# -*- coding: utf-8 -*-
"""
Unified fluorescence pipeline.

This script merges the workflow of four scripts into one orchestrator:
1) yolo_rayleigh_removal.py
2) pac_main.py
3) database_comparison.py
4) generate_ai_report.py

Each stage is wrapped as one class and executed in order.
All output paths are adapted to English naming.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StageResult:
    name: str
    success: bool
    message: str


class PathAdapter:
    """Migrate legacy Chinese output paths to English paths."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.moves = []

    def migrate(self):
        for src_rel, dst_rel in self.moves:
            src = self.repo_root / src_rel
            dst = self.repo_root / dst_rel
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.iterdir():
                    target = dst / item.name
                    try:
                        if item.is_file() and target.exists():
                            target.unlink()
                        elif item.is_dir() and target.exists():
                            shutil.rmtree(target)
                        shutil.move(str(item), str(target))
                    except FileNotFoundError:
                        # Source item may disappear during repeated migration runs.
                        continue
                try:
                    src.rmdir()
                except OSError:
                    pass
            else:
                try:
                    if dst.exists():
                        dst.unlink()
                    shutil.move(str(src), str(dst))
                except FileNotFoundError:
                    continue


class ScriptStageBase:
    def __init__(self, repo_root: Path, script_name: str, stage_name: str):
        self.repo_root = repo_root
        self.script_path = repo_root / "code" / script_name
        self.stage_name = stage_name

    def run(self) -> StageResult:
        if not self.script_path.exists():
            return StageResult(self.stage_name, False, f"script not found: {self.script_path}")

        cmd = [sys.executable, str(self.script_path)]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.repo_root / "code"),
                check=True,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
            )
            tail = completed.stdout[-1200:] if completed.stdout else ""
            return StageResult(self.stage_name, True, tail.strip() or "ok")
        except subprocess.CalledProcessError as exc:
            err = exc.stderr or exc.stdout or str(exc)
            return StageResult(self.stage_name, False, err[-1200:])


class RayleighRemovalStage(ScriptStageBase):
    def __init__(self, repo_root: Path):
        super().__init__(repo_root, "yolo_rayleigh_removal.py", "RayleighRemoval")


class PacParafacStage(ScriptStageBase):
    def __init__(self, repo_root: Path):
        super().__init__(repo_root, "pac_main.py", "PACParafac")


class DatabaseCompareStage(ScriptStageBase):
    def __init__(self, repo_root: Path):
        super().__init__(repo_root, "database_comparison.py", "DatabaseCompare")


class AIReportStage(ScriptStageBase):
    def __init__(self, repo_root: Path):
        super().__init__(repo_root, "generate_ai_report.py", "AIReport")


class FluorescencePipeline:
    """Run four stages in workflow order."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.path_adapter = PathAdapter(repo_root)
        self.stages = [
            RayleighRemovalStage(repo_root),
            PacParafacStage(repo_root),
            DatabaseCompareStage(repo_root),
            AIReportStage(repo_root),
        ]

    def run(self):
        print("[Pipeline] Start migration of legacy Chinese paths...")
        self.path_adapter.migrate()
        print("[Pipeline] Path migration done.")

        summary: list[StageResult] = []
        for stage in self.stages:
            print(f"\n[Pipeline] Running stage: {stage.stage_name}")
            result = stage.run()
            summary.append(result)
            print(f"[Pipeline] {stage.stage_name} success={result.success}")
            if result.message:
                print(result.message)

            if not result.success:
                print("[Pipeline] Stopped due to stage failure.")
                break

        print("\n[Pipeline] Summary")
        for r in summary:
            print(f"- {r.name}: {'OK' if r.success else 'FAILED'}")

        final_report = self.repo_root / "outputs" / "fluorescence_analysis_report.md"
        if final_report.exists():
            print(f"[Pipeline] Final report: {final_report}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(Path(__file__).resolve().parent)
    FluorescencePipeline(repo_root).run()
