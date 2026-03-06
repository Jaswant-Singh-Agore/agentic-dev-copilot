"""
Quick smoke test for the full copilot pipeline.
Run: python demo.py
"""

from pipeline.orchestrator import run_copilot

log_text = """
Traceback (most recent call last):
  File "utils.py", line 14, in calculate_average
    return total / count
ZeroDivisionError: division by zero
"""

code = """
def calculate_average(numbers: list) -> float:
    total = sum(numbers)
    count = len(numbers)
    return total / count
"""

result = run_copilot(task="full", log_text=log_text, code=code)
print(result["final_report"])