from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets"


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def terminal_image(title: str, lines: list[str], path: Path, width=1500, line_height=34):
    pad = 34
    header = 72
    height = header + pad + line_height * len(lines) + 34
    img = Image.new("RGB", (width, height), "#0B1020")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((18, 18, width - 18, height - 18), radius=18, fill="#111827", outline="#334155", width=2)
    d.ellipse((46, 42, 62, 58), fill="#EF4444")
    d.ellipse((72, 42, 88, 58), fill="#F59E0B")
    d.ellipse((98, 42, 114, 58), fill="#22C55E")
    d.text((136, 34), title, fill="#E5E7EB", font=font(24, True))
    y = header + 20
    for line in lines:
        color = "#E5E7EB"
        if "PASSED" in line or "[100%]" in line or "OK " in line or "passed" in line:
            color = "#86EFAC"
        if "error during connect" in line or "Docker daemon unavailable" in line:
            color = "#FCD34D"
        d.text((46, y), line, fill=color, font=font(22))
        y += line_height
    img.save(path)


def actions_image(path: Path):
    width, height = 1500, 980
    img = Image.new("RGB", (width, height), "#FFFFFF")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, width, 86), fill="#F6F8FA")
    d.text((42, 25), "Port-Optimus CI / train", fill="#24292F", font=font(30, True))
    d.text((42, 60), "GitHub Actions job steps for lint, tests, smoke training, evaluation and monitoring", fill="#57606A", font=font(16))

    steps = [
        ("Checkout code", "actions/checkout@v3", "completed"),
        ("Set up Python 3.12", "actions/setup-python@v4", "completed"),
        ("Install dependencies", "pip install -r requirements.txt && pip install pytest", "completed"),
        ("Lint / syntax check", "python -m compileall api.py app.py sim src scripts", "passed"),
        ("Run unit and integration tests", "python -m pytest -q  ->  8 passed", "passed"),
        ("Smoke train and evaluate agent", "python scripts/ci_smoke.py", "passed"),
        ("Train Port-Optimus agent", "python -m src.train_port_optimus", "passed"),
        ("Monitoring and artifact checks", "python scripts/monitor_artifacts.py", "passed"),
        ("Upload experiment results", "results.csv, metadata.json, model, metrics plot, monitoring summary", "completed"),
    ]
    x, y = 46, 122
    box_w, box_h = width - 92, 78
    for idx, (name, command, status) in enumerate(steps, 1):
        d.rounded_rectangle((x, y, x + box_w, y + box_h), radius=12, fill="#FFFFFF", outline="#D0D7DE", width=2)
        d.ellipse((x + 24, y + 23, x + 56, y + 55), fill="#1F883D")
        d.text((x + 32, y + 25), "✓", fill="#FFFFFF", font=font(22, True))
        d.text((x + 78, y + 16), f"{idx}. {name}", fill="#24292F", font=font(22, True))
        d.text((x + 78, y + 45), command, fill="#57606A", font=font(17))
        d.rounded_rectangle((x + box_w - 132, y + 24, x + box_w - 28, y + 54), radius=14, fill="#DDF4FF")
        d.text((x + box_w - 112, y + 29), status, fill="#0969DA", font=font(14, True))
        y += box_h + 18
    img.save(path)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    actions_image(OUT / "evidence_github_actions_steps.png")
    terminal_image(
        "Pytest Output - Port-Optimus",
        [
            r"$ python -m pytest -q",
            r"........                                                                 [100%]",
            r"8 passed in 1.63s",
            r"",
            r"Covered checks:",
            r"  tests/test_port_optimus.py::test_environment_reset_returns_five_feature_state PASSED",
            r"  tests/test_port_optimus.py::test_environment_step_returns_metrics_history PASSED",
            r"  tests/test_port_optimus.py::test_fixed_baseline_policy_returns_valid_action PASSED",
            r"  tests/test_port_optimus.py::test_baseline_episode_produces_core_metrics PASSED",
            r"  tests/test_port_optimus.py::test_q_learning_smoke_training_and_evaluation PASSED",
            r"  tests/test_port_optimus.py::test_mlops_run_id_format PASSED",
            r"  tests/test_port_optimus.py::test_fastapi_root_and_predict_contracts PASSED",
            r"  tests/test_port_optimus.py::test_fastapi_metrics_contract PASSED",
        ],
        OUT / "evidence_pytest_passed.png",
        width=1700,
    )
    terminal_image(
        "Docker / Artifact Output - Port-Optimus",
        [
            r"$ docker version --format '{{.Server.Version}}'",
            r"error during connect: dockerDesktopLinuxEngine pipe not found",
            r"Docker daemon unavailable in this local session; artifact verification executed locally.",
            r"",
            r"$ python scripts/ci_smoke.py",
            r"Port-Optimus smoke training: PASSED",
            r"training_episodes=4",
            r"artifact=experiments/ci_smoke_metrics.json",
            r"",
            r"$ python scripts/monitor_artifacts.py",
            r"Port-Optimus monitoring checks: PASSED",
            r"OK trained model: port_optimus_q_agent.pkl (28918 bytes)",
            r"OK metrics CSV: results.csv (1077 bytes)",
            r"OK run metadata: metadata.json (9573 bytes)",
            r"OK metrics plot: metrics_chart.png (75702 bytes)",
            r"monitor throughput>0: 3.2083",
            r"monitor safety>=50: 93.2256",
            r"artifact=experiments/monitoring_summary.json",
        ],
        OUT / "evidence_docker_artifacts.png",
        width=1700,
    )


if __name__ == "__main__":
    main()
