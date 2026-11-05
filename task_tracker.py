"""Small CodeSync indexing fixture: run with python task_tracker.py.
Change a method, upload the file, and inspect the updated chunks/vectors.
Uses only Python's standard library; no files or network are accessed.
"""
from dataclasses import dataclass
from datetime import date
import json

VALID_PRIORITIES = {"low": 1, "medium": 2, "high": 3}


@dataclass
class Task:
    """A task with a deadline, priority and completion state."""
    task_id: int
    title: str
    due_on: date
    priority: str = "medium"
    completed: bool = False

    def is_overdue(self, today: date) -> bool:
        """Completed tasks are never included in the overdue list."""
        return not self.completed and self.due_on < today

    def to_dict(self) -> dict:
        """Produce a JSON-compatible record for display."""
        return {"id": self.task_id, "title": self.title,
                "due_on": self.due_on.isoformat(),
                "priority": self.priority, "completed": self.completed}


class TaskTracker:
    """Store tasks in memory and provide filtering and reports."""

    def __init__(self) -> None:
        self.tasks: dict[int, Task] = {}
        self.next_id = 1

    def add_task(self, title: str, due_on: date, priority: str = "medium") -> Task:
        """Validate inputs before assigning a new task identifier."""
        title = title.strip()
        if not title:
            raise ValueError("Task title cannot be empty")
        if priority not in VALID_PRIORITIES:
            raise ValueError("Priority must be low, medium or high")
        task = Task(self.next_id, title, due_on, priority)
        self.tasks[task.task_id] = task
        self.next_id += 1
        return task

    def complete_task(self, task_id: int) -> None:
        """Mark an existing task complete; unknown identifiers raise KeyError."""
        self.tasks[task_id].completed = True

    def pending_tasks(self) -> list[Task]:
        """Sort pending work by descending priority, then earliest deadline."""
        def sort_key(task: Task) -> tuple:
            return (-VALID_PRIORITIES[task.priority], task.due_on, task.task_id)
        return sorted((t for t in self.tasks.values() if not t.completed),
                      key=sort_key)

    def overdue_tasks(self, today: date) -> list[Task]:
        """Return pending tasks whose deadline has passed."""
        return [task for task in self.pending_tasks() if task.is_overdue(today)]

    def search(self, phrase: str) -> list[Task]:
        """Find titles containing a case-insensitive search phrase."""
        return [t for t in self.tasks.values() if phrase.casefold() in t.title.casefold()]

    def report(self, today: date) -> dict:
        """Calculate totals without changing any task state."""
        total = len(self.tasks)
        done = sum(task.completed for task in self.tasks.values())
        return {"total": total, "completed": done, "pending": total - done,
                "overdue": len(self.overdue_tasks(today)),
                "completion_percent": round(done * 100 / total, 1) if total else 0.0}


def build_demo() -> TaskTracker:
    """Create deterministic sample data for repeatable demonstrations."""
    tracker = TaskTracker()
    tracker.add_task("Connect GitHub webhook", date(2026, 11, 3), "high")
    tracker.add_task("Inspect code embeddings", date(2026, 11, 6), "medium")
    tracker.add_task("Update project notes", date(2026, 11, 7), "low")
    tracker.complete_task(3)
    return tracker


def main() -> None:
    """Print the report and ordered tasks without external dependencies."""
    tracker = build_demo()
    today = date(2026, 11, 5)
    assert tracker.report(today)["overdue"] == 1
    assert tracker.pending_tasks()[0].task_id == 1
    assert tracker.search("EMBEDDINGS")[0].task_id == 2
    print(json.dumps({"report": tracker.report(today),
                      "tasks": [t.to_dict() for t in tracker.pending_tasks()]}, indent=2))

if __name__ == "__main__":
    main()
