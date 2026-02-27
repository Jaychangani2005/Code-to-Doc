# s_tool\exceptions.py

## Overview
The `s_tool\exceptions.py` module defines custom exception classes for handling errors in the Selenium tool. It provides a base class `SToolException` and a specific exception class `InvalidWebDriverError` to handle invalid WebDriver instances. These exceptions can be raised and caught in the Selenium tool to handle and report errors effectively.

## Dependencies
This module does not import any internal or external libraries.

## Classes

### SToolException
Base class for Selenium tool exceptions.

*   Description: This class serves as a base class for all exceptions raised in the Selenium tool. It provides a way to handle and report errors in a standardized manner.
*   Key Methods:
    *   `__init__(self, message)`: Initializes the exception with a custom error message.
    *   `__str__(self)`: Returns the custom error message as a string.

### InvalidWebDriverError
Custom exception for an invalid WebDriver instance.

*   Description: This exception is raised when an invalid WebDriver instance is encountered.
*   Key Methods:
    *   `__init__(self)`: Initializes the exception with a default error message.

## Functions
None

## Usage Example
```python
try:
    # Code that may raise an exception
    raise SToolException("Invalid WebDriver instance")
except SToolException as e:
    # Handle the exception
    print(f"Error: {e}")
```
In this example, we raise an `SToolException` with a custom error message and catch it to handle the error. The `__str__` method of the exception class is used to get the error message as a string.