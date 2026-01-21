# Task App Module

This module defines a set of Celery tasks for background processing. It includes simple arithmetic operations, a blocking task that simulates a long-running process, and a task that reports its progress.

## Architecture

This module directly uses Celery's `shared_task` decorator to register functions as asynchronous tasks. It interacts with the Celery worker to execute these tasks independently of the main application flow.

## Functions

*   `add(a: int, b: int) -> int`: A simple Celery task that adds two integers and returns the result.
*   `block() -> None`: A Celery task that simulates a blocking operation by sleeping for 5 seconds.
*   `process(self: Task, total: int) -> object`: A Celery task that simulates a long-running process, updating its state with progress information periodically.