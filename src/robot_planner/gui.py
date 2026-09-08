"""Polished obstacle-aware desktop simulator for Robot Path Lab.

Dataset generation remains outside this module.  The GUI consumes the shared
Phase 4 single-trajectory evaluator, so the values shown on screen and exported
from the simulator use exactly the same mathematics as the generated dataset.
Creating a Tk window happens only in :func:`main`, keeping imports headless-safe.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Mapping

from .curves import CURVE_LABELS, CurvePath, make_curve
from .evaluation import CircleObstacle
from .kinematics import JointSolution, PlanarArm2R
from .phase4_evaluation import MathematicalTrajectoryResult, simulate_trajectory


BG = "#F3F6FB"
PANEL = "#FFFFFF"
INK = "#172033"
MUTED = "#667085"
BORDER = "#DDE3EE"
PRIMARY = "#3563E9"
PRIMARY_DARK = "#244BC5"
TEAL = "#10A6A1"
ORANGE = "#F59E0B"
GREEN = "#149C62"
RED = "#D64545"
GRID = "#E9EDF5"
FONT = "Malgun Gothic"
SAMPLES = 161


def _number(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(slots=True)
class ViewTransform:
    world_left: float
    world_bottom: float
    scale: float
    canvas_height: float

    def map(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            (point[0] - self.world_left) * self.scale,
            self.canvas_height - (point[1] - self.world_bottom) * self.scale,
        )


class PlannerApp:
    """Interactive laboratory for one selected 2R path and obstacle scene."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Robot Path Lab · Mathematical 2R Simulator")
        self.root.geometry("1540x900")
        self.root.minsize(1180, 740)
        self.root.configure(bg=BG)

        self.curve: CurvePath | None = None
        self.arm: PlanarArm2R | None = None
        self.simulation: MathematicalTrajectoryResult | None = None
        self.solutions: list[JointSolution | None] = []
        self.obstacles: list[CircleObstacle] = []
        self._obstacle_serial = 0
        self.animating = False
        self._animation_job: str | None = None

        self._build_style()
        self._build_variables()
        self._build_layout()
        self._recompute(show_errors=False)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Clean.TCombobox",
            fieldbackground=PANEL,
            background=PANEL,
            foreground=INK,
            bordercolor=BORDER,
            arrowcolor=PRIMARY,
            padding=7,
            font=(FONT, 9),
        )
        style.map(
            "Clean.TCombobox",
            fieldbackground=[("readonly", PANEL)],
            selectbackground=[("readonly", PANEL)],
            selectforeground=[("readonly", INK)],
        )
        style.configure(
            "Clean.Horizontal.TScale",
            background=PANEL,
            troughcolor="#E5EAF3",
            bordercolor=PANEL,
            lightcolor=PRIMARY,
            darkcolor=PRIMARY,
        )
        style.configure("Clean.TNotebook", background=PANEL, borderwidth=0)
        style.configure(
            "Clean.TNotebook.Tab",
            font=(FONT, 8, "bold"),
            padding=(12, 7),
            background="#EEF2F8",
            foreground=MUTED,
        )
        style.map(
            "Clean.TNotebook.Tab",
            background=[("selected", PANEL)],
            foreground=[("selected", PRIMARY_DARK)],
        )

    def _build_variables(self) -> None:
        self.base_x = tk.StringVar(value="0.0")
        self.base_y = tk.StringVar(value="0.0")
        self.link1 = tk.StringVar(value="3.2")
        self.link2 = tk.StringVar(value="2.6")
        self.start_x = tk.StringVar(value="1.5")
        self.start_y = tk.StringVar(value="2.0")
        self.end_x = tk.StringVar(value="4.3")
        self.end_y = tk.StringVar(value="1.1")
        self.duration = tk.StringVar(value="3.0")
        self.curve_kind = tk.StringVar(value="3차 베지어")
        self.elbow_mode = tk.StringVar(value="팔꿈치 위")
        self.shape = tk.DoubleVar(value=0.38)
        self.shape_readout = tk.StringVar(value="+0.38")
        self.animation_speed = tk.DoubleVar(value=1.0)
        self.animation_speed_readout = tk.StringVar(value="×1.00")
        self.pose_index = tk.DoubleVar(value=0.0)
        self.obstacle_x = tk.StringVar(value="2.7")
        self.obstacle_y = tk.StringVar(value="1.2")
        self.obstacle_radius = tk.StringVar(value="0.45")

        self.status_text = tk.StringVar(value="계산 준비")
        self.kpi_status = tk.StringVar(value="—")
        self.kpi_length = tk.StringVar(value="—")
        self.kpi_clearance = tk.StringVar(value="—")
        self.live_time = tk.StringVar(value="t = 0.000 s · u = 0.0000")
        self.live_pose = tk.StringVar(value="θ₁ = — · θ₂ = — · 장애물 없음")
        self.current_sample = tk.StringVar(value="현재 표본 정보가 여기에 표시됩니다.")
        self.metric_vars = {
            key: tk.StringVar(value="—")
            for key in (
                "path_length_units",
                "joint_velocity_effort",
                "joint_acceleration_effort",
                "joint_jerk_effort",
                "regularized_curvature_energy_inv_unit",
                "obstacle_penalty_s_per_unit2",
            )
        }

    @staticmethod
    def _panel(parent: tk.Misc, **kwargs: object) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
            **kwargs,
        )

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg="#15213A", height=72)
        header.pack(fill="x")
        header.pack_propagate(False)
        title = tk.Frame(header, bg="#15213A")
        title.pack(side="left", padx=27, pady=11)
        tk.Label(
            title,
            text="ROBOT PATH LAB",
            bg="#15213A",
            fg="white",
            font=(FONT, 16, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title,
            text="2관절 로봇팔 · 장애물 · 수학적 경로 평가",
            bg="#15213A",
            fg="#AEBBD6",
            font=(FONT, 9),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="2R  ·  2D  ·  DATA LAB",
            bg="#22345B",
            fg="#DCE7FF",
            font=("Segoe UI", 9, "bold"),
            padx=13,
            pady=7,
        ).pack(side="right", padx=27)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=16)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, minsize=330)
        body.grid_columnconfigure(1, weight=1, minsize=490)
        body.grid_columnconfigure(2, minsize=370)

        sidebar = self._panel(body, width=330)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 11))
        sidebar.grid_propagate(False)
        self._build_sidebar(sidebar)

        center = tk.Frame(body, bg=BG)
        center.grid(row=0, column=1, sticky="nsew", padx=(0, 11))
        center.grid_rowconfigure(1, weight=1)
        center.grid_columnconfigure(0, weight=1)
        self._build_center(center)

        details = self._panel(body, width=370)
        details.grid(row=0, column=2, sticky="nsew")
        details.grid_propagate(False)
        self._build_details(details)

    def _scrollable(self, parent: tk.Frame) -> tk.Frame:
        canvas = tk.Canvas(parent, bg=PANEL, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=PANEL)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window, width=event.width),
        )

        def on_wheel(event: tk.Event[tk.Misc]) -> None:
            canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", on_wheel))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        return content

    @staticmethod
    def _heading(parent: tk.Misc, title: str, subtitle: str = "") -> None:
        tk.Label(parent, text=title, bg=PANEL, fg=INK, font=(FONT, 10, "bold")).pack(anchor="w")
        if subtitle:
            tk.Label(parent, text=subtitle, bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(
                anchor="w", pady=(2, 0)
            )

    @staticmethod
    def _separator(parent: tk.Misc) -> None:
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=12)

    def _entry_pair(
        self,
        parent: tk.Misc,
        label: str,
        first: tk.StringVar,
        second: tk.StringVar,
        first_hint: str = "x",
        second_hint: str = "y",
    ) -> None:
        tk.Label(parent, text=label, bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(
            anchor="w", pady=(7, 4)
        )
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x")
        for index, (variable, hint) in enumerate(
            ((first, first_hint), (second, second_hint))
        ):
            box = tk.Frame(row, bg="#F8FAFD", highlightbackground=BORDER, highlightthickness=1)
            box.pack(side="left", fill="x", expand=True, padx=(0, 5) if index == 0 else (5, 0))
            tk.Label(
                box,
                text=hint,
                bg="#F8FAFD",
                fg="#98A2B3",
                font=("Segoe UI", 8, "bold"),
                width=2,
            ).pack(side="left", padx=(4, 0))
            tk.Entry(
                box,
                textvariable=variable,
                relief="flat",
                bd=0,
                bg="#F8FAFD",
                fg=INK,
                font=("Segoe UI", 9),
                width=7,
            ).pack(side="left", fill="x", expand=True, padx=4, pady=6)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        content = tk.Frame(self._scrollable(parent), bg=PANEL)
        content.pack(fill="both", expand=True, padx=18, pady=16)
        self._heading(content, "실험 조건", "좌표, 팔 길이와 이동시간")
        self._entry_pair(content, "로봇 밑부분", self.base_x, self.base_y)
        self._entry_pair(content, "팔 길이", self.link1, self.link2, "L₁", "L₂")
        self._entry_pair(content, "끝점 시작 위치", self.start_x, self.start_y)
        self._entry_pair(content, "끝점 도착 위치", self.end_x, self.end_y)
        time_row = tk.Frame(content, bg=PANEL)
        time_row.pack(fill="x", pady=(8, 0))
        tk.Label(time_row, text="이동시간", bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(side="left")
        tk.Label(time_row, text="초", bg=PANEL, fg="#98A2B3", font=(FONT, 8)).pack(side="right")
        tk.Entry(
            time_row,
            textvariable=self.duration,
            relief="flat",
            bd=0,
            bg="#F8FAFD",
            highlightbackground=BORDER,
            highlightthickness=1,
            font=("Segoe UI", 9),
            width=8,
        ).pack(side="right", padx=6, ipady=5)

        self._separator(content)
        self._heading(content, "경로 모델", "선택한 곡선 하나를 시뮬레이션")
        tk.Label(content, text="곡선 종류", bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(
            anchor="w", pady=(8, 4)
        )
        curve_box = ttk.Combobox(
            content,
            textvariable=self.curve_kind,
            values=list(CURVE_LABELS),
            state="readonly",
            style="Clean.TCombobox",
        )
        curve_box.pack(fill="x")
        curve_box.bind("<<ComboboxSelected>>", lambda _event: self._recompute())
        tk.Label(content, text="역기구학 해", bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(
            anchor="w", pady=(8, 4)
        )
        elbow_box = ttk.Combobox(
            content,
            textvariable=self.elbow_mode,
            values=["팔꿈치 위", "팔꿈치 아래"],
            state="readonly",
            style="Clean.TCombobox",
        )
        elbow_box.pack(fill="x")
        elbow_box.bind("<<ComboboxSelected>>", lambda _event: self._recompute())
        shape_row = tk.Frame(content, bg=PANEL)
        shape_row.pack(fill="x", pady=(10, 0))
        tk.Label(shape_row, text="곡선 굽힘", bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(side="left")
        tk.Label(
            shape_row,
            textvariable=self.shape_readout,
            bg="#EEF3FF",
            fg=PRIMARY_DARK,
            font=("Consolas", 8, "bold"),
            padx=7,
            pady=2,
        ).pack(side="right")
        ttk.Scale(
            content,
            from_=-1.0,
            to=1.0,
            variable=self.shape,
            command=lambda value: self.shape_readout.set(f"{float(value):+.2f}"),
            style="Clean.Horizontal.TScale",
        ).pack(fill="x", pady=(4, 3))

        self._separator(content)
        self._heading(content, "원형 장애물", "중심 (x,y), 반지름 r · 여러 개 가능")
        obstacle_row = tk.Frame(content, bg=PANEL)
        obstacle_row.pack(fill="x", pady=(8, 5))
        for column, (label, variable) in enumerate(
            (("x", self.obstacle_x), ("y", self.obstacle_y), ("r", self.obstacle_radius))
        ):
            obstacle_row.grid_columnconfigure(column, weight=1)
            box = tk.Frame(obstacle_row, bg="#F8FAFD", highlightbackground=BORDER, highlightthickness=1)
            box.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 3, 0 if column == 2 else 3))
            tk.Label(box, text=label, bg="#F8FAFD", fg="#98A2B3", font=("Segoe UI", 8, "bold")).pack(
                side="left", padx=(4, 0)
            )
            tk.Entry(box, textvariable=variable, relief="flat", bd=0, bg="#F8FAFD", width=5).pack(
                side="left", fill="x", expand=True, padx=2, pady=6
            )
        buttons = tk.Frame(content, bg=PANEL)
        buttons.pack(fill="x")
        self._mini_button(buttons, "+ 추가", self._add_obstacle, PRIMARY).pack(
            side="left", fill="x", expand=True, padx=(0, 3)
        )
        self._mini_button(buttons, "선택 삭제", self._remove_obstacle, "#6B7892").pack(
            side="left", fill="x", expand=True, padx=3
        )
        self._mini_button(buttons, "전체 삭제", self._clear_obstacles, "#98A2B3").pack(
            side="left", fill="x", expand=True, padx=(3, 0)
        )
        self.obstacle_listbox = tk.Listbox(
            content,
            height=4,
            relief="flat",
            bd=0,
            bg="#F7F9FC",
            fg=INK,
            selectbackground="#DDE7FF",
            selectforeground=PRIMARY_DARK,
            highlightbackground=BORDER,
            highlightthickness=1,
            font=("Consolas", 8),
            activestyle="none",
        )
        self.obstacle_listbox.pack(fill="x", pady=(6, 0))

        self._separator(content)
        speed_title = tk.Frame(content, bg=PANEL)
        speed_title.pack(fill="x")
        tk.Label(speed_title, text="재생 속도", bg=PANEL, fg=INK, font=(FONT, 10, "bold")).pack(side="left")
        tk.Label(
            speed_title,
            textvariable=self.animation_speed_readout,
            bg="#EEF3FF",
            fg=PRIMARY_DARK,
            font=("Consolas", 8, "bold"),
            padx=7,
            pady=2,
        ).pack(side="right")
        ttk.Scale(
            content,
            from_=0.25,
            to=4.0,
            variable=self.animation_speed,
            command=lambda value: self.animation_speed_readout.set(f"×{float(value):.2f}"),
            style="Clean.Horizontal.TScale",
        ).pack(fill="x", pady=(7, 2))
        tk.Label(
            content,
            text="화면 재생 속도만 변경하며 평가용 T는 유지됩니다.",
            bg=PANEL,
            fg="#98A2B3",
            font=(FONT, 7),
        ).pack(anchor="w")
        tk.Button(
            content,
            text="경로 다시 계산",
            command=self._recompute,
            bg=PRIMARY,
            activebackground=PRIMARY_DARK,
            activeforeground="white",
            fg="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            font=(FONT, 9, "bold"),
            pady=9,
        ).pack(fill="x", pady=(12, 2))

    @staticmethod
    def _mini_button(
        parent: tk.Misc,
        text: str,
        command: Callable[[], None],
        color: str,
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            activebackground=color,
            activeforeground="white",
            fg="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            font=(FONT, 7, "bold"),
            pady=5,
        )

    def _build_center(self, parent: tk.Frame) -> None:
        kpis = tk.Frame(parent, bg=BG)
        kpis.grid(row=0, column=0, sticky="ew", pady=(0, 11))
        for column in range(3):
            kpis.grid_columnconfigure(column, weight=1)
        self._kpi(kpis, 0, "경로 유효성", self.kpi_status, TEAL)
        self._kpi(kpis, 1, "끝점 궤적 길이 L", self.kpi_length, PRIMARY)
        self._kpi(kpis, 2, "최소 장애물 여유", self.kpi_clearance, ORANGE)

        workspace = self._panel(parent)
        workspace.grid(row=1, column=0, sticky="nsew")
        workspace.grid_rowconfigure(1, weight=1)
        workspace.grid_columnconfigure(0, weight=1)
        titlebar = tk.Frame(workspace, bg=PANEL)
        titlebar.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 8))
        tk.Label(titlebar, text="장애물 작업 공간", bg=PANEL, fg=INK, font=(FONT, 10, "bold")).pack(
            side="left"
        )
        self.workspace_status = tk.Label(
            titlebar,
            textvariable=self.status_text,
            bg="#ECF8F3",
            fg=GREEN,
            font=(FONT, 8, "bold"),
            padx=9,
            pady=4,
        )
        self.workspace_status.pack(side="right")
        self.workspace_canvas = tk.Canvas(workspace, bg="#FBFCFE", highlightthickness=0, bd=0)
        self.workspace_canvas.grid(row=1, column=0, sticky="nsew", padx=11)
        self.workspace_canvas.bind("<Configure>", lambda _event: self._draw_workspace())

        live = tk.Frame(workspace, bg="#F7F9FC", highlightbackground=BORDER, highlightthickness=1)
        live.grid(row=2, column=0, sticky="ew", padx=16, pady=(9, 0))
        tk.Label(live, textvariable=self.live_time, bg="#F7F9FC", fg=INK, font=("Consolas", 9, "bold")).pack(
            anchor="w", padx=10, pady=(7, 1)
        )
        tk.Label(live, textvariable=self.live_pose, bg="#F7F9FC", fg=MUTED, font=(FONT, 8)).pack(
            anchor="w", padx=10, pady=(1, 7)
        )

        controls = tk.Frame(workspace, bg=PANEL)
        controls.grid(row=3, column=0, sticky="ew", padx=16, pady=9)
        self.animate_button = tk.Button(
            controls,
            text="▶ 재생",
            command=self._toggle_animation,
            bg="#EEF3FF",
            activebackground="#E1E9FF",
            fg=PRIMARY_DARK,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=(FONT, 8, "bold"),
            padx=10,
            pady=6,
        )
        self.animate_button.pack(side="left", padx=(0, 5))
        tk.Button(
            controls,
            text="↺ 처음",
            command=self._reset_animation,
            bg="#F3F5F9",
            activebackground="#E8ECF3",
            fg=MUTED,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=(FONT, 8, "bold"),
            padx=9,
            pady=6,
        ).pack(side="left", padx=(0, 8))
        self.pose_scale = ttk.Scale(
            controls,
            from_=0,
            to=SAMPLES - 1,
            variable=self.pose_index,
            command=lambda _value: self._draw_workspace(),
            style="Clean.Horizontal.TScale",
        )
        self.pose_scale.pack(side="left", fill="x", expand=True)
        self.pose_readout = tk.Label(
            controls,
            text="0.000 s",
            bg=PANEL,
            fg=MUTED,
            font=("Consolas", 8),
            width=9,
        )
        self.pose_readout.pack(side="right", padx=(8, 0))

        exports = tk.Frame(workspace, bg=PANEL)
        exports.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 10))
        tk.Label(exports, text="현재 실행 저장", bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(side="left")
        self._export_button(exports, "JSON", self._export_json).pack(side="right", padx=(5, 0))
        self._export_button(exports, "CSV", self._export_csv).pack(side="right")

    def _kpi(
        self,
        parent: tk.Misc,
        column: int,
        label: str,
        variable: tk.StringVar,
        accent: str,
    ) -> None:
        card = self._panel(parent)
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 4, 0 if column == 2 else 4))
        tk.Frame(card, width=4, bg=accent).pack(side="left", fill="y")
        text = tk.Frame(card, bg=PANEL)
        text.pack(side="left", padx=11, pady=9)
        tk.Label(text, text=label, bg=PANEL, fg=MUTED, font=(FONT, 7)).pack(anchor="w")
        tk.Label(text, textvariable=variable, bg=PANEL, fg=INK, font=(FONT, 9, "bold")).pack(
            anchor="w", pady=(2, 0)
        )

    @staticmethod
    def _export_button(parent: tk.Misc, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            parent,
            text=f"⇩ {text}",
            command=command,
            bg="#F3F5F9",
            activebackground="#E8ECF3",
            fg=INK,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=(FONT, 8, "bold"),
            padx=10,
            pady=5,
        )

    def _build_details(self, parent: tk.Frame) -> None:
        heading = tk.Frame(parent, bg=PANEL)
        heading.pack(fill="x", padx=16, pady=(15, 8))
        self._heading(heading, "수학적 해석", "곡선식, θ(t) 합성식과 수치적분")
        tabs = ttk.Notebook(parent, style="Clean.TNotebook")
        tabs.pack(fill="both", expand=True, padx=11, pady=(0, 11))
        formula = tk.Frame(tabs, bg=PANEL)
        metrics = tk.Frame(tabs, bg=PANEL)
        chart = tk.Frame(tabs, bg=PANEL)
        tabs.add(formula, text="수식")
        tabs.add(metrics, text="평가값")
        tabs.add(chart, text="관절각")

        tk.Label(formula, text="곡선식과 최소 저크 시간함수", bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(
            anchor="w", padx=8, pady=(10, 4)
        )
        self.equation_text = self._readonly_text(formula, 9)
        self.equation_text.pack(fill="x", padx=8)
        tk.Label(formula, text="역기구학으로 합성한 θ₁(t), θ₂(t)", bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(
            anchor="w", padx=8, pady=(10, 4)
        )
        self.ik_text = self._readonly_text(formula, 10)
        self.ik_text.pack(fill="x", padx=8)
        tk.Label(formula, text="곡선 매개변수", bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(
            anchor="w", padx=8, pady=(10, 4)
        )
        self.parameter_label = tk.Label(
            formula,
            text="—",
            justify="left",
            anchor="nw",
            bg="#F7F9FC",
            fg=INK,
            font=("Consolas", 8),
            padx=9,
            pady=7,
        )
        self.parameter_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tk.Label(
            metrics,
            text="모든 값은 작을수록 좋은 비용 함수입니다.",
            bg="#EEF3FF",
            fg=PRIMARY_DARK,
            font=(FONT, 8),
            padx=9,
            pady=7,
        ).pack(fill="x", padx=8, pady=(10, 8))
        specs = (
            ("L", "경로 길이", "path_length_units", "unit"),
            ("Eθ", "관절 운동 노력도 (실제 J 아님)", "joint_velocity_effort", "rad²/s"),
            ("A", "각가속도 제곱 적분", "joint_acceleration_effort", "rad²/s³"),
            ("J", "각저크 제곱 적분", "joint_jerk_effort", "rad²/s⁵"),
            (
                "Kε",
                "정칙화 곡률 비용",
                "regularized_curvature_energy_inv_unit",
                "1/unit",
            ),
            ("O", "장애물 접근 벌점", "obstacle_penalty_s_per_unit2", "s/unit²"),
        )
        for symbol, label, key, unit in specs:
            row = tk.Frame(metrics, bg=PANEL)
            row.pack(fill="x", padx=8, pady=4)
            tk.Label(
                row,
                text=symbol,
                width=3,
                bg="#EEF3FF",
                fg=PRIMARY_DARK,
                font=("Cambria Math", 10, "bold"),
                padx=4,
                pady=6,
            ).pack(side="left")
            tk.Label(row, text=label, bg=PANEL, fg=INK, font=(FONT, 8)).pack(side="left", padx=8)
            tk.Label(row, text=unit, bg=PANEL, fg="#98A2B3", font=(FONT, 7)).pack(side="right")
            tk.Label(row, textvariable=self.metric_vars[key], bg=PANEL, fg=INK, font=("Consolas", 8, "bold")).pack(
                side="right", padx=5
            )
        current = tk.Frame(metrics, bg="#F7F9FC", highlightbackground=BORDER, highlightthickness=1)
        current.pack(fill="x", padx=8, pady=(13, 8))
        tk.Label(current, text="현재 표본", bg="#F7F9FC", fg=INK, font=(FONT, 8, "bold")).pack(
            anchor="w", padx=9, pady=(7, 2)
        )
        tk.Label(
            current,
            textvariable=self.current_sample,
            justify="left",
            wraplength=315,
            bg="#F7F9FC",
            fg=MUTED,
            font=(FONT, 8),
        ).pack(anchor="w", padx=9, pady=(1, 8))
        tk.Label(
            metrics,
            text=(
                "θ̇, θ̈, θ⃛은 7점 유한차분, 적분은 합성 사다리꼴 공식으로 계산합니다. "
                "충돌 검사는 161개 시간표본에서 수행하므로 매우 작은 표본 사이 충돌은 놓칠 수 있습니다."
            ),
            justify="left",
            wraplength=320,
            bg="#FFF8E7",
            fg="#845B06",
            font=(FONT, 8),
            padx=9,
            pady=8,
        ).pack(fill="x", padx=8, pady=(5, 8))

        legend = tk.Frame(chart, bg=PANEL)
        legend.pack(fill="x", padx=8, pady=(10, 3))
        tk.Label(legend, text="시간에 따른 관절각", bg=PANEL, fg=INK, font=(FONT, 9, "bold")).pack(side="left")
        self._legend(legend, "θ₁", PRIMARY)
        self._legend(legend, "θ₂", TEAL)
        self.angle_canvas = tk.Canvas(chart, bg="#FBFCFE", highlightthickness=0, height=330)
        self.angle_canvas.pack(fill="both", expand=True, padx=8, pady=(6, 8))
        self.angle_canvas.bind("<Configure>", lambda _event: self._draw_angle_chart())

    @staticmethod
    def _readonly_text(parent: tk.Misc, height: int) -> tk.Text:
        widget = tk.Text(
            parent,
            height=height,
            wrap="word",
            bg="#F7F9FC",
            fg=INK,
            relief="flat",
            bd=0,
            highlightbackground=BORDER,
            highlightthickness=1,
            font=(FONT, 8),
            padx=9,
            pady=8,
        )
        widget.configure(state="disabled")
        return widget

    @staticmethod
    def _legend(parent: tk.Misc, label: str, color: str) -> None:
        box = tk.Frame(parent, bg=PANEL)
        box.pack(side="right", padx=(10, 0))
        marker = tk.Canvas(box, width=10, height=10, bg=PANEL, highlightthickness=0)
        marker.pack(side="left")
        marker.create_oval(2, 2, 8, 8, fill=color, outline=color)
        tk.Label(box, text=label, bg=PANEL, fg=MUTED, font=(FONT, 8)).pack(side="left")

    @staticmethod
    def _read_float(variable: tk.StringVar, name: str, *, positive: bool = False) -> float:
        try:
            value = float(variable.get())
        except ValueError as error:
            raise ValueError(f"{name}에는 숫자를 입력해야 합니다.") from error
        if not math.isfinite(value):
            raise ValueError(f"{name}에는 유한한 숫자를 입력해야 합니다.")
        if positive and value <= 0.0:
            raise ValueError(f"{name}은(는) 0보다 커야 합니다.")
        return value

    def _add_obstacle(self) -> None:
        try:
            x = self._read_float(self.obstacle_x, "장애물 x")
            y = self._read_float(self.obstacle_y, "장애물 y")
            radius = self._read_float(self.obstacle_radius, "장애물 반지름", positive=True)
        except ValueError as error:
            messagebox.showerror("장애물 입력 확인", str(error), parent=self.root)
            return
        self._obstacle_serial += 1
        self.obstacles.append(CircleObstacle(f"O{self._obstacle_serial}", x, y, radius))
        self._refresh_obstacles()
        self._recompute()

    def _remove_obstacle(self) -> None:
        for index in reversed(self.obstacle_listbox.curselection()):
            del self.obstacles[index]
        self._refresh_obstacles()
        self._recompute()

    def _clear_obstacles(self) -> None:
        self.obstacles.clear()
        self._refresh_obstacles()
        self._recompute()

    def _refresh_obstacles(self) -> None:
        self.obstacle_listbox.delete(0, "end")
        for obstacle in self.obstacles:
            self.obstacle_listbox.insert(
                "end",
                f"{obstacle.obstacle_id}: ({obstacle.center_x:.3g}, {obstacle.center_y:.3g}), r={obstacle.radius_units:.3g}",
            )

    def _recompute(self, *, show_errors: bool = True) -> None:
        self._stop_animation()
        try:
            base = (self._read_float(self.base_x, "밑부분 x"), self._read_float(self.base_y, "밑부분 y"))
            start = (self._read_float(self.start_x, "시작점 x"), self._read_float(self.start_y, "시작점 y"))
            end = (self._read_float(self.end_x, "도착점 x"), self._read_float(self.end_y, "도착점 y"))
            arm = PlanarArm2R(
                self._read_float(self.link1, "첫 번째 팔 길이", positive=True),
                self._read_float(self.link2, "두 번째 팔 길이", positive=True),
                base,
            )
            curve = make_curve(self.curve_kind.get(), start, end, shape=self.shape.get(), samples=SAMPLES)
            duration = self._read_float(self.duration, "이동시간", positive=True)
            mode = "up" if self.elbow_mode.get() == "팔꿈치 위" else "down"
            result = simulate_trajectory(
                arm,
                curve,
                mode,
                obstacles=tuple(self.obstacles),
                duration_s=duration,
                samples=SAMPLES,
            )
        except (ValueError, KeyError, TypeError) as error:
            if show_errors:
                messagebox.showerror("입력 확인", str(error), parent=self.root)
            return

        self.arm, self.curve, self.simulation = arm, curve, result
        self.solutions = []
        for row in result.sample_rows:
            theta1 = _number(row, "theta1_rad")
            theta2 = _number(row, "theta2_rad")
            self.solutions.append(
                None if theta1 is None or theta2 is None else JointSolution(theta1, theta2, mode)  # type: ignore[arg-type]
            )

        metrics = result.metrics
        self.kpi_length.set(self._format(_number(metrics, "path_length_units"), " unit"))
        clearance = _number(metrics, "min_obstacle_clearance_units")
        self.kpi_clearance.set("장애물 없음" if not self.obstacles else self._format(clearance, " unit"))
        reachable = bool(metrics["fully_reachable"])
        collision_free = metrics["collision_free"] is not False
        if not reachable:
            self.kpi_status.set("도달 불가능")
            self.status_text.set("도달 불가능 구간 있음")
            self.workspace_status.configure(bg="#FFF0F0", fg=RED)
        elif not collision_free:
            self.kpi_status.set("충돌 발생")
            self.status_text.set("표본에서 장애물 충돌")
            self.workspace_status.configure(bg="#FFF0F0", fg=RED)
        else:
            self.kpi_status.set("안전한 유효 경로")
            self.status_text.set("전체 도달 · 표본 충돌 없음")
            self.workspace_status.configure(bg="#ECF8F3", fg=GREEN)
        self.pose_index.set(0.0)
        self.pose_scale.configure(to=len(result.sample_rows) - 1)
        self._update_details()
        self._draw_workspace()
        self._draw_angle_chart()

    @staticmethod
    def _format(value: float | None, suffix: str = "") -> str:
        if value is None:
            return "—"
        if value != 0.0 and (abs(value) >= 100_000.0 or abs(value) < 0.001):
            return f"{value:.3e}{suffix}"
        return f"{value:.5f}{suffix}"

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _update_details(self) -> None:
        if self.curve is None or self.simulation is None:
            return
        eq = self.simulation.equations
        self._set_text(
            self.equation_text,
            "\n".join(
                (
                    eq["curve_equation_template"],
                    eq["x_u_equation"],
                    eq["y_u_equation"],
                    "",
                    eq["progress_equation"],
                    eq["x_t_equation"],
                    eq["y_t_equation"],
                )
            ),
        )
        self._set_text(
            self.ik_text,
            "\n\n".join(
                (
                    eq["ik_cosine_equation"],
                    eq["theta2_t_equation"],
                    eq["theta1_t_equation"],
                    f"분기·연속성: {eq['branch_policy']}",
                )
            ),
        )
        parameters = []
        for key, value in self.curve.parameters.items():
            parameters.append(f"{key:<15} = {value: .5f}" if isinstance(value, float) else f"{key:<15} = {value}")
        self.parameter_label.configure(text="\n".join(parameters) if parameters else "추가 매개변수 없음")
        for key, variable in self.metric_vars.items():
            value = _number(self.simulation.metrics, key)
            variable.set(self._format(value))

    @staticmethod
    def _grid_step(span: float) -> float:
        target = max(span / 8.0, 1e-9)
        power = 10.0 ** math.floor(math.log10(target))
        scaled = target / power
        return (1.0 if scaled < 1.8 else 2.0 if scaled < 3.8 else 5.0) * power

    def _transform(self, width: float, height: float) -> ViewTransform | None:
        if self.arm is None or self.curve is None or width < 20 or height < 20:
            return None
        bx, by = self.arm.base
        reach = self.arm.max_reach
        xs = [bx - reach, bx + reach, *(point[0] for point in self.curve.points)]
        ys = [by - reach, by + reach, *(point[1] for point in self.curve.points)]
        for obstacle in self.obstacles:
            xs.extend((obstacle.center_x - obstacle.radius_units, obstacle.center_x + obstacle.radius_units))
            ys.extend((obstacle.center_y - obstacle.radius_units, obstacle.center_y + obstacle.radius_units))
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        span_x, span_y = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
        min_x, max_x = min_x - 0.12 * span_x, max_x + 0.12 * span_x
        min_y, max_y = min_y - 0.12 * span_y, max_y + 0.12 * span_y
        scale = min(width / (max_x - min_x), height / (max_y - min_y))
        min_x -= (width - (max_x - min_x) * scale) / (2.0 * scale)
        min_y -= (height - (max_y - min_y) * scale) / (2.0 * scale)
        return ViewTransform(min_x, min_y, scale, height)

    def _draw_workspace(self) -> None:
        canvas = self.workspace_canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 2), max(canvas.winfo_height(), 2)
        transform = self._transform(width, height)
        if transform is None or self.arm is None or self.curve is None or self.simulation is None:
            return
        span_x, span_y = width / transform.scale, height / transform.scale
        left, bottom = transform.world_left, transform.world_bottom
        right, top = left + span_x, bottom + span_y
        step = self._grid_step(max(span_x, span_y))
        x = math.floor(left / step) * step
        while x <= right:
            px, _ = transform.map((x, 0.0))
            canvas.create_line(px, 0, px, height, fill=GRID)
            x += step
        y = math.floor(bottom / step) * step
        while y <= top:
            _, py = transform.map((0.0, y))
            canvas.create_line(0, py, width, py, fill=GRID)
            y += step
        base = transform.map(self.arm.base)
        for radius, dash in ((self.arm.max_reach, (5, 4)), (self.arm.min_reach, (3, 4))):
            pixels = radius * transform.scale
            canvas.create_oval(base[0] - pixels, base[1] - pixels, base[0] + pixels, base[1] + pixels, outline="#AFBAD0", dash=dash)
        for obstacle in self.obstacles:
            cx, cy = transform.map(obstacle.center)
            radius = obstacle.radius_units * transform.scale
            canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill="#FDECEC", outline=RED, width=2)
            canvas.create_text(cx, cy, text=obstacle.obstacle_id, fill="#A52F2F", font=("Segoe UI", 7, "bold"))
        coordinates = [value for point in self.curve.points for value in transform.map(point)]
        canvas.create_line(*coordinates, fill=PRIMARY, width=4, smooth=True, capstyle="round")

        rows = self.simulation.sample_rows
        for left_row, right_row in zip(rows, rows[1:]):
            if bool(left_row["collision_at_sample"]) or bool(right_row["collision_at_sample"]):
                canvas.create_line(
                    *transform.map((float(left_row["x"]), float(left_row["y"]))),
                    *transform.map((float(right_row["x"]), float(right_row["y"]))),
                    fill=RED,
                    width=5,
                )
        ghost_indexes = {0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4, len(rows) - 1}
        for index in ghost_indexes:
            if self.solutions[index] is not None:
                self._draw_arm(canvas, transform, self.solutions[index], ghost=True, collision=False)
        index = max(0, min(int(round(self.pose_index.get())), len(rows) - 1))
        if self.solutions[index] is not None:
            self._draw_arm(
                canvas,
                transform,
                self.solutions[index],
                ghost=False,
                collision=bool(rows[index]["collision_at_sample"]),
            )
        else:
            px, py = transform.map((float(rows[index]["x"]), float(rows[index]["y"])))
            canvas.create_line(px - 6, py - 6, px + 6, py + 6, fill=RED, width=2)
            canvas.create_line(px - 6, py + 6, px + 6, py - 6, fill=RED, width=2)
        self._point(canvas, transform, self.curve.start, TEAL, "START")
        self._point(canvas, transform, self.curve.end, ORANGE, "GOAL")
        self._update_live(index)

    def _draw_arm(
        self,
        canvas: tk.Canvas,
        transform: ViewTransform,
        solution: JointSolution,
        *,
        ghost: bool,
        collision: bool,
    ) -> None:
        assert self.arm is not None
        pose = self.arm.forward(solution.theta1, solution.theta2)
        base, elbow, end = transform.map(pose.base), transform.map(pose.elbow), transform.map(pose.end)
        if ghost:
            canvas.create_line(*base, *elbow, fill="#C8D2E5", width=2)
            canvas.create_line(*elbow, *end, fill="#C8D2E5", width=2)
            return
        canvas.create_line(*base, *elbow, fill=RED if collision else "#263B63", width=8, capstyle="round")
        canvas.create_line(*elbow, *end, fill=RED if collision else TEAL, width=7, capstyle="round")
        for point, fill, radius in ((base, "#15213A", 7), (elbow, ORANGE, 7), (end, PRIMARY, 6)):
            canvas.create_oval(point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius, fill=fill, outline="white", width=2)

    @staticmethod
    def _point(canvas: tk.Canvas, transform: ViewTransform, point: tuple[float, float], color: str, label: str) -> None:
        x, y = transform.map(point)
        canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=PANEL, outline=color, width=3)
        canvas.create_text(x, y - 14, text=label, fill=color, font=("Segoe UI", 7, "bold"))

    def _update_live(self, index: int) -> None:
        assert self.simulation is not None
        row = self.simulation.sample_rows[index]
        time_s, progress = float(row["time_s"]), float(row["progress_u"])
        self.pose_readout.configure(text=f"{time_s:.3f} s")
        self.live_time.set(f"t = {time_s:.3f} s · u = {progress:.4f} · 표본 {index + 1}/{len(self.simulation.sample_rows)}")
        theta1, theta2 = _number(row, "theta1_rad"), _number(row, "theta2_rad")
        if theta1 is None or theta2 is None:
            self.live_pose.set("θ₁ = — · θ₂ = — · 도달 불가능")
            self.current_sample.set(f"t={time_s:.3f} s, u={progress:.4f}\n역기구학 해가 없습니다.")
            return
        clearance = _number(row, "obstacle_clearance")
        obstacle_text = "장애물 없음"
        if self.obstacles:
            obstacle_text = "여유 계산 불가" if clearance is None else f"여유 {clearance:.4f} unit"
            if bool(row["collision_at_sample"]):
                obstacle_text += " · 충돌"
        self.live_pose.set(
            f"θ₁ = {theta1:.4f} rad ({math.degrees(theta1):.2f}°) · θ₂ = {theta2:.4f} rad ({math.degrees(theta2):.2f}°) · {obstacle_text}"
        )
        self.current_sample.set(
            f"t={time_s:.3f} s, u={progress:.4f}\nθ₁(t)={theta1:.6f} rad\nθ₂(t)={theta2:.6f} rad\n"
            + ("이 표본에서 충돌" if bool(row["collision_at_sample"]) else "도달 가능 · 표본 충돌 없음")
        )

    def _draw_angle_chart(self) -> None:
        canvas = self.angle_canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 2), max(canvas.winfo_height(), 2)
        if self.simulation is None:
            return
        rows = self.simulation.sample_rows
        angles = [float(row[key]) for row in rows for key in ("theta1_deg", "theta2_deg") if row[key] is not None]
        if not angles:
            canvas.create_text(width / 2, height / 2, text="표시할 관절각이 없습니다", fill=MUTED, font=(FONT, 9))
            return
        left, right, top, bottom = 40.0, 10.0, 12.0, 25.0
        plot_width, plot_height = max(width - left - right, 1.0), max(height - top - bottom, 1.0)
        lower = math.floor((min(angles) - 10.0) / 30.0) * 30.0
        upper = math.ceil((max(angles) + 10.0) / 30.0) * 30.0
        if upper <= lower:
            upper = lower + 60.0
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + fraction * plot_height
            canvas.create_line(left, y, width - right, y, fill=GRID)
            canvas.create_text(left - 5, y, text=f"{upper - fraction * (upper - lower):.0f}°", anchor="e", fill="#98A2B3", font=("Segoe UI", 7))
        canvas.create_text(left, height - 6, text="t=0", anchor="w", fill="#98A2B3", font=("Segoe UI", 7))
        canvas.create_text(width - right, height - 6, text=f"t={float(rows[-1]['time_s']):g}s", anchor="e", fill="#98A2B3", font=("Segoe UI", 7))
        for key, color in (("theta1_deg", PRIMARY), ("theta2_deg", TEAL)):
            segments: list[list[float]] = []
            current: list[float] = []
            for index, row in enumerate(rows):
                value = row[key]
                if value is None:
                    if len(current) >= 4:
                        segments.append(current)
                    current = []
                    continue
                x = left + index / (len(rows) - 1) * plot_width
                y = top + (upper - float(value)) / (upper - lower) * plot_height
                current.extend((x, y))
            if len(current) >= 4:
                segments.append(current)
            for coordinates in segments:
                canvas.create_line(*coordinates, fill=color, width=2, smooth=True)

    def _toggle_animation(self) -> None:
        if self.animating:
            self._stop_animation()
            return
        if self.simulation is None:
            return
        if self.pose_index.get() >= len(self.simulation.sample_rows) - 1:
            self.pose_index.set(0.0)
        self.animating = True
        self.animate_button.configure(text="Ⅱ 일시정지")
        self._animate_step()

    def _animate_step(self) -> None:
        if not self.animating or self.simulation is None:
            return
        index = int(round(self.pose_index.get()))
        if index >= len(self.simulation.sample_rows) - 1:
            self._stop_animation()
            return
        self.pose_index.set(index + 1)
        self._draw_workspace()
        duration = float(self.simulation.metrics["duration_s"])
        speed = max(self.animation_speed.get(), 0.01)
        delay = max(8, int(1000 * duration / (len(self.simulation.sample_rows) - 1) / speed))
        self._animation_job = self.root.after(delay, self._animate_step)

    def _reset_animation(self) -> None:
        self._stop_animation()
        self.pose_index.set(0.0)
        self._draw_workspace()

    def _stop_animation(self) -> None:
        self.animating = False
        if hasattr(self, "animate_button"):
            self.animate_button.configure(text="▶ 재생")
        if self._animation_job is not None:
            try:
                self.root.after_cancel(self._animation_job)
            except tk.TclError:
                pass
            self._animation_job = None

    def _payload(self) -> dict[str, object] | None:
        if self.arm is None or self.curve is None or self.simulation is None:
            return None
        mode = "up" if self.elbow_mode.get() == "팔꿈치 위" else "down"
        return {
            "export_format_version": "robot-path-single-run-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "base_x": self.arm.base[0],
                "base_y": self.arm.base[1],
                "link1": self.arm.link1,
                "link2": self.arm.link2,
                "start_x": self.curve.start[0],
                "start_y": self.curve.start[1],
                "goal_x": self.curve.end[0],
                "goal_y": self.curve.end[1],
                "duration_s": self.simulation.metrics["duration_s"],
                "curve_key": self.curve.key,
                "curve_name": self.curve.display_name,
                "shape": self.shape.get(),
                "elbow_mode": mode,
                "samples": len(self.simulation.sample_rows),
            },
            "obstacles": [obstacle.as_dict() for obstacle in self.obstacles],
            "curve_parameters": dict(self.curve.parameters),
            "equations": self.simulation.equations,
            "metrics": self.simulation.metrics,
            "samples": list(self.simulation.sample_rows),
        }

    def _export_json(self) -> None:
        payload = self._payload()
        if payload is None:
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="현재 시뮬레이션 JSON 저장",
            initialfile="robot_path_run.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("모든 파일", "*.*")],
        )
        if not filename:
            return
        try:
            Path(filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except (OSError, ValueError) as error:
            messagebox.showerror("저장 실패", str(error), parent=self.root)
            return
        messagebox.showinfo("저장 완료", filename, parent=self.root)

    def _export_csv(self) -> None:
        payload = self._payload()
        if payload is None or self.simulation is None:
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="현재 시뮬레이션 CSV 저장",
            initialfile="robot_path_samples.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("모든 파일", "*.*")],
        )
        if not filename:
            return
        inputs = payload["inputs"]
        assert isinstance(inputs, Mapping)
        metric_keys = (
            "path_length_units",
            "joint_velocity_effort",
            "joint_acceleration_effort",
            "joint_jerk_effort",
            "regularized_curvature_energy_inv_unit",
            "obstacle_penalty_s_per_unit2",
        )
        common: dict[str, object] = {
            key: inputs[key]
            for key in ("curve_key", "curve_name", "shape", "elbow_mode", "duration_s")
        }
        common.update({key: self.simulation.metrics[key] for key in metric_keys})
        common.update(
            {
                "selection_valid": self.simulation.metrics["selection_valid"],
                "warning": self.simulation.metrics["warning"],
                "curve_equation": self.simulation.equations["curve_equation"],
                "x_t_equation": self.simulation.equations["x_t_equation"],
                "y_t_equation": self.simulation.equations["y_t_equation"],
                "theta1_t_equation": self.simulation.equations["theta1_t_equation"],
                "theta2_t_equation": self.simulation.equations["theta2_t_equation"],
                "equation_format_version": self.simulation.equations[
                    "equation_format_version"
                ],
                "obstacles_json": json.dumps(
                    [obstacle.as_dict() for obstacle in self.obstacles],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        rows = [{**common, **row} for row in self.simulation.sample_rows]
        fieldnames = [*common, *self.simulation.sample_rows[0]]
        try:
            with Path(filename).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
        except OSError as error:
            messagebox.showerror("저장 실패", str(error), parent=self.root)
            return
        messagebox.showinfo("저장 완료", filename, parent=self.root)


def main() -> None:
    root = tk.Tk()
    app = PlannerApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop_animation(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
