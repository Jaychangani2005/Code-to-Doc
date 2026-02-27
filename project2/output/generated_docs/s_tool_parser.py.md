# Lxml Parser Module
=====================

## Overview
------------

The `s_tool\parser.py` module provides a set of parser utilities using the lxml library. It offers a class-based approach to parsing HTML strings, making it easier to extract data from web pages. The module is designed to be flexible and extensible, allowing users to write their own custom parsers.

## Dependencies
------------

### Internal Dependencies

None

### External Libraries Used

*   `lxml.html`: A library for parsing and manipulating HTML documents.

## Classes
---------

### LxmlParser

The `LxmlParser` class is the primary parser utility in this module. It provides two methods for parsing HTML strings:

#### `dropdown`

*   **Purpose:** Parse a dropdown from an HTML string and return the options as a list of tuples.
*   **Parameters:**
    *   `html_string`: The HTML string containing the dropdown element.
    *   `text_exclude`: A list of values to exclude from the result (optional).
*   **Return Value:** A list of tuples containing key-value pairs of the dropdown options.

#### `table`

*   **Purpose:** Write your own custom parser class.
*   **Parameters:** None
*   **Return Value:** Raises a `NotImplementedError` exception, indicating that the user should write their own custom parser.

## Functions
------------

None

## Usage Example
--------------

```markdown
### Example Usage

```python
from s_tool.parser import LxmlParser

# Create an instance of the LxmlParser class
parser = LxmlParser()

# Define an HTML string containing a dropdown element
html_string = """
<select>
    <option value="option1">Option 1</option>
    <option value="option2">Option 2</option>
    <option value="option3">Option 3</option>
</select>
"""

# Parse the dropdown using the parser
result = parser.dropdown(html_string)

# Print the result
print(result)
```

This code snippet demonstrates how to use the `LxmlParser` class to parse a dropdown from an HTML string. The `dropdown` method is used to extract the options as a list of tuples, which are then printed to the console.