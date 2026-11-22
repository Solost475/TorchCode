"""Core judge engine — grabs user functions from the Jupyter namespace and tests them."""

from __future__ import annotations

import time
import traceback
from typing import Any

from torch_judge.tasks import DIFFICULTY_LABELS, get_task, TASKS
from torch_judge.progress import mark_solved, mark_attempted

_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_DIM = "\033[90m"
_BOLD = "\033[1m"


def _get_user_namespace() -> dict[str, Any]:
    """Get the calling notebook's global namespace via IPython."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            return ip.user_ns
    except ImportError:
        pass
    # Fallback: try to grab caller's globals
    import inspect
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        return frame.f_back.f_back.f_globals
    return {}


def check(task_id: str) -> None:
    """Run all tests for a task against the user's implementation.

    Usage in a notebook cell:
        from torch_judge import check
        check("relu")
    """
    task = get_task(task_id)
    if task is None:
        available = ", ".join(f"'{k}'" for k in TASKS)
        print(f"{_RED}未知题目 '{task_id}'。可用题目：{available}{_RESET}")
        return

    fn_name = task["function_name"]
    user_ns = _get_user_namespace()

    if fn_name not in user_ns:
        print(f"\n{_RED}❌ 笔记本中未找到函数或类 '{fn_name}'。{_RESET}")
        print(f"{_DIM}   请先在上方单元格中定义它，并运行该单元格。{_RESET}\n")
        return

    user_fn = user_ns[fn_name]
    tests = task["tests"]
    passed = 0
    total = len(tests)

    difficulty = DIFFICULTY_LABELS.get(task["difficulty"], task["difficulty"])
    print(f"\n{_BOLD}🧪 正在测试：{task['title']}（{difficulty}）{_RESET}")
    print(f"{'─' * 50}")

    total_time = 0.0

    for i, test in enumerate(tests, 1):
        test_code = test["code"].replace("{fn}", fn_name)
        namespace: dict[str, Any] = {fn_name: user_fn}

        t0 = time.perf_counter()
        try:
            exec(compile(test_code, f"<test:{test['name']}>", "exec"), namespace)  # noqa: S102
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            passed += 1
            print(f"  {_GREEN}✅ [{i}/{total}] {test['name']}{_RESET} {_DIM}({elapsed*1000:.1f}ms){_RESET}")
        except AssertionError as e:
            elapsed = time.perf_counter() - t0
            msg = str(e) or "断言失败"
            print(f"  {_RED}❌ [{i}/{total}] {test['name']}{_RESET}")
            print(f"     {_RED}{msg}{_RESET}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  {_RED}💥 [{i}/{total}] {test['name']}{_RESET}")
            print(f"     {_RED}{type(e).__name__}: {e}{_RESET}")
            tb = traceback.format_exc()
            short_tb = "\n".join(tb.strip().split("\n")[-3:])
            print(f"     {_DIM}{short_tb}{_RESET}")

    print(f"{'─' * 50}")

    if passed == total:
        print(f"  {_GREEN}{_BOLD}🎉 全部 {total} 个测试通过！（总耗时 {total_time*1000:.1f}ms）{_RESET}")
        mark_solved(task_id, total_time)
        print(f"  {_DIM}进度已保存。运行 status() 查看练习进度。{_RESET}\n")
    else:
        print(f"  {_YELLOW}📊 已通过 {passed}/{total} 个测试。{_RESET}")
        mark_attempted(task_id)
        print(f"  {_DIM}继续加油！遇到困难时可运行 hint(\"{task_id}\") 查看提示。{_RESET}\n")


def hint(task_id: str) -> None:
    """Show a hint for the given task."""
    task = get_task(task_id)
    if task is None:
        print(f"{_RED}未知题目 '{task_id}'。{_RESET}")
        return
    print(f"\n{_YELLOW}💡 {task['title']} 的提示：{_RESET}")
    print(f"   {task['hint']}\n")
