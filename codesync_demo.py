"""Runnable library-management fixture for CodeSync indexing (Python 3.10+).

Run: python codesync_demo.py
Check: python codesync_demo.py --self-test
Uses only the standard library; no network access or persistent writes.
For a sync demo, change calculate_late_fee(), commit, and push the file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from functools import wraps
from typing import Callable, Iterator

DAILY_LATE_FEE_CENTS = 250
DEFAULT_LOAN_DAYS = 14
MAX_ACTIVE_LOANS = 3


class LibraryError(ValueError):
    """An invalid library operation that callers can report to users."""


def validate_identifier(value: str) -> str:
    """Normalize an identifier and reject empty or whitespace-containing IDs."""
    normalized = value.strip().upper()
    if not normalized or any(char.isspace() for char in normalized):
        raise LibraryError("Identifiers must be nonempty and contain no spaces")
    return normalized


def audited(action: str) -> Callable:
    """Decorate a service method to record successful mutations only."""
    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            result = function(self, *args, **kwargs)
            self.audit_log.append({"action": action, "method": function.__name__})
            return result
        return wrapper
    return decorate


@dataclass(frozen=True)
class Book:
    """Immutable catalog entry; stock is derived from active loans."""
    book_id: str
    title: str
    author: str
    tags: tuple[str, ...] = ()
    copies: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "book_id", validate_identifier(self.book_id))
        if self.copies < 1 or not self.title.strip() or not self.author.strip():
            raise LibraryError("A book needs a title, author and positive stock")

    @property
    def searchable_text(self) -> str:
        """Combine descriptive fields for case-insensitive catalog search."""
        return " ".join((self.title, self.author, *self.tags)).casefold()

    @classmethod
    def from_mapping(cls, data: dict) -> Book:
        """Build a typed book record from JSON-like input."""
        return cls(
            book_id=str(data["book_id"]),
            title=str(data["title"]),
            author=str(data["author"]),
            tags=tuple(data.get("tags", ())),
            copies=int(data.get("copies", 1)),
        )


@dataclass(frozen=True)
class Member:
    """A registered borrower with a normalized identifier."""
    member_id: str
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_id", validate_identifier(self.member_id))
        if not self.name.strip():
            raise LibraryError("Member name is required")


@dataclass
class Loan:
    """One borrowing transaction with an optional return date."""
    loan_id: str
    book_id: str
    member_id: str
    borrowed_on: date
    due_on: date
    returned_on: date | None = None

    @property
    def active(self) -> bool:
        return self.returned_on is None

    def overdue_days(self, as_of: date) -> int:
        """Stop accruing late days when the book is returned."""
        effective_date = self.returned_on or as_of
        return max(0, (effective_date - self.due_on).days)

    def to_record(self) -> dict:
        """Convert dates into JSON-compatible ISO strings."""
        record = asdict(self)
        for field in ("borrowed_on", "due_on", "returned_on"):
            value = record[field]
            record[field] = value.isoformat() if value else None
        return record


def calculate_late_fee(days: int, daily_cents: int = DAILY_LATE_FEE_CENTS) -> int:
    """Return an integer fee in cents without floating-point currency errors."""
    if days < 0 or daily_cents < 0:
        raise LibraryError("Days and daily fee cannot be negative")
    return days * daily_cents


class LibraryService:
    """Coordinate catalog search, borrowing, returns and reporting in memory."""

    def __init__(self) -> None:
        self.books: dict[str, Book] = {}
        self.members: dict[str, Member] = {}
        self.loans: dict[str, Loan] = {}
        self.audit_log: list[dict[str, str]] = []

    @audited("catalog.add")
    def add_book(self, book: Book) -> None:
        if book.book_id in self.books:
            raise LibraryError("Book already exists")
        self.books[book.book_id] = book

    @audited("member.register")
    def register_member(self, member: Member) -> None:
        if member.member_id in self.members:
            raise LibraryError("Member already exists")
        self.members[member.member_id] = member

    def active_loans(self, member_id: str | None = None) -> Iterator[Loan]:
        """Yield active loans, optionally restricted to one borrower."""
        scope = validate_identifier(member_id) if member_id is not None else None
        for loan in self.loans.values():
            if loan.active and (scope is None or loan.member_id == scope):
                yield loan

    def available_copies(self, book_id: str) -> int:
        book = self.books[validate_identifier(book_id)]
        borrowed = sum(loan.book_id == book.book_id for loan in self.active_loans())
        return book.copies - borrowed

    def search_books(self, query: str) -> list[Book]:
        """Match all query tokens and sort matching titles predictably."""
        tokens = query.casefold().split()
        def matches(book: Book) -> bool:
            return all(token in book.searchable_text for token in tokens)
        return sorted(filter(matches, self.books.values()), key=lambda b: b.title)

    @audited("loan.borrow")
    def borrow_book(self, book_id: str, member_id: str, today: date) -> Loan:
        book_id = validate_identifier(book_id)
        member_id = validate_identifier(member_id)
        if book_id not in self.books or member_id not in self.members:
            raise LibraryError("Unknown book or member")
        if self.available_copies(book_id) < 1:
            raise LibraryError("No copies available")
        if sum(1 for _ in self.active_loans(member_id)) >= MAX_ACTIVE_LOANS:
            raise LibraryError("Member has reached the borrowing limit")
        loan_id = f"L{len(self.loans) + 1:04d}"
        loan = Loan(loan_id, book_id, member_id, today,
                    today + timedelta(days=DEFAULT_LOAN_DAYS))
        self.loans[loan_id] = loan
        return loan

    @audited("loan.return")
    def return_book(self, loan_id: str, today: date) -> int:
        """Close an active loan and return its assessed late fee in cents."""
        loan_id = validate_identifier(loan_id)
        if loan_id not in self.loans:
            raise LibraryError("Unknown loan")
        loan = self.loans[loan_id]
        if not loan.active:
            raise LibraryError("Loan has already been returned")
        if today < loan.borrowed_on:
            raise LibraryError("Return cannot precede borrowing")
        fee = calculate_late_fee(loan.overdue_days(today))
        loan.returned_on = today
        return fee

    def overdue_report(self, today: date) -> list[dict]:
        """List only open overdue loans with their outstanding late fees."""
        return [
            {**loan.to_record(), "fee_cents": calculate_late_fee(days)}
            for loan in self.active_loans()
            if (days := loan.overdue_days(today)) > 0
        ]

    async def inventory_snapshot(self) -> dict:
        """Yield control, then return an in-memory inventory summary."""
        await asyncio.sleep(0)
        tags = Counter(tag for book in self.books.values() for tag in book.tags)
        return {
            "titles": len(self.books),
            "members": len(self.members),
            "active_loans": sum(1 for _ in self.active_loans()),
            "available": {key: self.available_copies(key) for key in self.books},
            "tags": dict(sorted(tags.items())),
        }


def build_demo_library() -> LibraryService:
    """Create deterministic seed records without external files or services."""
    service = LibraryService()
    rows = [
        ("B001", "Python Patterns", "A. Rao", ("python", "design"), 2),
        ("B002", "Practical Git", "M. Shah", ("git", "version-control"), 1),
        ("B003", "Search Systems", "R. Das", ("search", "indexing"), 1),
    ]
    for row in rows:
        service.add_book(Book(*row))
    service.register_member(Member("M001", "Demo Reader"))
    return service


def run_self_tests() -> None:
    """Exercise stock, search, fees, validation, audit and async behavior."""
    service = build_demo_library()
    today = date(2026, 11, 3)
    loan = service.borrow_book("b002", "m001", today)
    assert service.available_copies("B002") == 0
    audit_count = len(service.audit_log)
    try:
        service.borrow_book("B002", "M001", today)
    except LibraryError:
        pass
    else:
        raise AssertionError("Out-of-stock borrowing should fail")
    assert len(service.audit_log) == audit_count
    assert service.search_books("python design")[0].book_id == "B001"
    assert service.search_books("missing") == []
    late_date = loan.due_on + timedelta(days=3)
    assert service.overdue_report(late_date)[0]["fee_cents"] == 750
    assert service.return_book(loan.loan_id, late_date) == 750
    assert service.available_copies("B002") == 1
    assert service.overdue_report(late_date) == []
    assert loan.overdue_days(late_date + timedelta(days=20)) == 3
    assert asyncio.run(service.inventory_snapshot())["active_loans"] == 0
    assert Book.from_mapping(asdict(service.books["B001"])) == service.books["B001"]
    print("All self-tests passed.")


async def run_demo() -> None:
    """Print a repeatable JSON report suitable for checking the fixture."""
    service = build_demo_library()
    today = date(2026, 11, 3)
    loan = service.borrow_book("B001", "M001", today)
    report = {
        "inventory": await service.inventory_snapshot(),
        "loan": loan.to_record(),
        "overdue": service.overdue_report(today + timedelta(days=17)),
        "audit": service.audit_log,
    }
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run built-in checks")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
    else:
        asyncio.run(run_demo())


if __name__ == "__main__":
    main()
