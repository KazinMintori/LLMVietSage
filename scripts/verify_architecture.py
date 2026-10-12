"""Audit every first-party source file against the generation architecture."""
import argparse
import ast
import json
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("app", "config", "scripts", "tests", "training", "proactive_recommendation")
FORBIDDEN_RUNTIME_IMPORTS = (
    "sqlalchemy", "alembic", "chromadb", "pgvector", "pypdf",
    "app.api", "app.adapters", "app.models",
)
ALLOWED_ROUTES = {"/info", "/health/live", "/chat", "/chat/stream"}


def _module(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(tree: ast.AST, module: str, package: bool) -> list[str]:
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported = node.module or ""
            if node.level:
                base = module.split(".")
                if not package:
                    base.pop()
                base = base[:len(base) - node.level + 1]
                imported = ".".join(base + imported.split(".")) if imported else ".".join(base)
            if imported:
                modules.append(imported)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def _inventory(root: Path) -> list[dict]:
    """Classify tracked/unignored artifacts; never read runtime env or corpora."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True, check=False,
    )
    paths = []
    if result.returncode == 0:
        paths = [root / name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    else:
        paths = list(root.rglob("*"))
    paths.extend(root.glob(".env*"))
    inventory = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        top = relative.parts[0]
        if top in {".git", "runtime", ".venv", "venv", "__pycache__"} or "__pycache__" in relative.parts:
            continue
        if top == "training" and any(part in {"raw", "prepared", "models", "artifacts"} for part in relative.parts[1:-1]):
            if path.name != ".gitkeep":
                continue
        if path.name.startswith(".env"):
            category = "root_environment"
        elif path.suffix == ".py":
            category = "source"
        elif top in {"datasets", "evals"} or top == "training" and "data" in relative.parts:
            category = "offline_data_or_evaluation"
        elif top == "proactive_recommendation":
            category = "independent_research"
        elif path.suffix == ".md":
            category = "documentation"
        else:
            category = "deployment_or_support"
        inventory.append({"file": str(relative), "category": category, "status": "classified"})
    return inventory


def audit_project(root: Path = ROOT, config_baseline: dict | None = None) -> dict:
    root = root.resolve()
    failures = []
    files = []
    graphs = {}
    models = {}
    routes = set()

    def fail(path, rule, line=0):
        failures.append({"file": str(path.relative_to(root)), "line": line, "rule": rule})

    for folder in SOURCE_ROOTS:
        for path in sorted((root / folder).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root)
            if ".venv" in relative.parts or "venv" in relative.parts:
                continue
            if relative.parts[0] == "training" and any(
                part in {"raw", "prepared", "models", "artifacts"} for part in relative.parts[1:-1]
            ):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except (SyntaxError, UnicodeError) as exc:
                fail(path, f"Invalid source: {type(exc).__name__}", getattr(exc, "lineno", 0))
                continue
            scope = path.relative_to(root).parts[0]
            module = _module(path, root)
            imports = _imports(tree, module, path.name == "__init__.py")
            graphs[module] = imports
            models[module] = path
            files.append({"file": str(path.relative_to(root)), "scope": scope, "status": "checked"})
            if scope != "app":
                continue
            layer = path.relative_to(root / "app").parts[0]
            if layer not in {"main.py", "__init__.py", "services", "controllers", "core", "domain"}:
                fail(path, "Unexpected runtime layer")
            if path.name != "__init__.py":
                if layer == "services" and not path.stem.endswith("_service"):
                    fail(path, "Service filename must end in _service.py")
                if layer == "controllers" and not path.stem.endswith("_controller"):
                    fail(path, "Controller filename must end in _controller.py")
                if "schema" in path.relative_to(root / "app").parts and not path.stem.endswith("_schema"):
                    fail(path, "Schema filename must end in _schema.py")
            for imported in imports:
                if imported.startswith(FORBIDDEN_RUNTIME_IMPORTS):
                    fail(path, f"Out-of-scope runtime dependency: {imported}")
                if layer == "domain" and imported.startswith(
                    ("fastapi", "httpx", "config", "app.core", "app.controllers", "app.services")
                ):
                    fail(path, f"Domain cannot depend on {imported}")
                if layer == "core" and imported.startswith("app.") and not imported.startswith("app.core"):
                    fail(path, f"Core must stay independent of {imported}")
                if layer == "services" and imported.startswith(("fastapi", "starlette", "app.controllers")):
                    fail(path, f"Service cannot depend on HTTP transport: {imported}")
                if layer == "controllers" and imported.startswith(("httpx", "config")):
                    fail(path, f"Controller cannot call provider/config directly: {imported}")
            for node in ast.walk(tree):
                if layer != "services" and isinstance(node, (ast.If, ast.IfExp, ast.Match, ast.BoolOp)):
                    fail(path, "Conditional decisions must be in services", node.lineno)
                if layer != "services" and isinstance(node, ast.comprehension) and node.ifs:
                    fail(path, "Conditional comprehension must be in services", node.iter.lineno)
                if layer == "controllers" and isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Try)):
                    fail(path, "Controller must remain transport-only", node.lineno)
                if isinstance(node, ast.ClassDef):
                    decorators = [
                        (d.func if isinstance(d, ast.Call) else d)
                        for d in node.decorator_list
                    ]
                    if any(isinstance(d, ast.Name) and d.id == "dataclass" for d in decorators):
                        if path.parent.name != "models":
                            fail(path, "Data records must be in domain/models", node.lineno)
                    if any(isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases):
                        if path.parent.name != "schema":
                            fail(path, "Pydantic schemas must be in domain/schema", node.lineno)
                if isinstance(node, ast.Call):
                    if layer == "controllers" and isinstance(node.func, ast.Name) and node.func.id == "APIRouter":
                        prefix = next((kw.value for kw in node.keywords if kw.arg == "prefix"), None)
                        if not isinstance(prefix, ast.Constant) or prefix.value != "/api":
                            fail(path, "Inference route prefix must be /api", node.lineno)
                    method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
                    if method in {"write_text", "write_bytes", "mkdir", "unlink", "save", "commit"}:
                        fail(path, "Inference must not persist caller data", node.lineno)
                    if method in {"get", "post", "put", "delete", "patch"} and layer == "controllers":
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "router":
                            route = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                            routes.add(route)
                            if route not in ALLOWED_ROUTES:
                                fail(path, f"Route outside allowed routes: {route}", node.lineno)
                    if method in {"embed", "embed_batch", "embeddings"}:
                        fail(path, "Embedding belongs to upstream Data Service", node.lineno)

    for module, imports in graphs.items():
        for imported in imports:
            if imported.startswith(("app.", "config.")) and imported not in graphs:
                fail(models[module], f"Unresolved first-party import: {imported}")

    visited = set()
    def visit(module, stack):
        if module in stack:
            fail(models[module], "Import cycle: " + " -> ".join(stack + [module]))
            return
        if module in visited:
            return
        for imported in graphs.get(module, []):
            if imported in graphs:
                visit(imported, stack + [module])
        visited.add(module)
    for module in graphs:
        visit(module, [])

    for path in (root / "app").rglob("*.pyc"):
        name = path.name.split(".")[0] + ".py"
        if not (path.parent.parent / name).exists():
            fail(path, "Orphaned bytecode from removed/renamed runtime module")
    if (root / "legacy").exists():
        fail(root / "legacy", "Legacy directory must be removed")
    if (root / "app").exists() and routes != ALLOWED_ROUTES:
        fail(root / "app", "Runtime route inventory does not match allowed routes")
    for path in (root / "config").rglob(".env*"):
        fail(path, "Runtime environment files belong at project root")
    config_hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "config").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    if config_baseline is not None:
        normalized_baseline = {name.replace("\\", "/"): digest.lower() for name, digest in config_baseline.items()}
        if config_hashes != normalized_baseline:
            fail(root / "config", "Protected config files changed from the task baseline")
    failed_files = {item["file"] for item in failures}
    for item in files:
        if item["file"] in failed_files:
            item["status"] = "failed"
    return {
        "passed": not failures,
        "source_files": len(files),
        "runtime_files": sum(item["scope"] == "app" for item in files),
        "routes": sorted(route for route in routes if isinstance(route, str)),
        "files": files, "artifacts": _inventory(root),
        "config_hashes": config_hashes, "violations": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--config-baseline", type=Path)
    args = parser.parse_args()
    baseline = None
    if args.config_baseline:
        baseline = json.loads(args.config_baseline.read_text(encoding="utf-8"))
    report = audit_project(config_baseline=baseline)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in report.items() if key not in {"files", "artifacts", "config_hashes"}
    }, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
