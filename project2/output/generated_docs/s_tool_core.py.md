# SeleniumTools

## Overview
SeleniumTools is a utility class that provides various Selenium-related functions for automating web browsers. It allows users to create a Selenium WebDriver instance, perform actions on web elements, and interact with web pages. The class also provides methods for parsing HTML elements, attaching custom parsers, and managing cookies.

## Dependencies

### Internal Dependencies
- `exceptions`: This module contains custom exceptions for SeleniumTools.
- `parser`: This module contains a parser class for parsing HTML elements.
- `logger`: This module contains a logger class for logging events.
- `driver`: This module contains a driver class for creating a Selenium WebDriver instance.

### External Libraries Used
- `selenium.webdriver.support.select`: This library provides support for selecting elements using the Select class.
- `selenium.common.exceptions`: This library provides common exceptions for Selenium.
- `urllib.parse`: This library provides functions for parsing URLs.
- `selenium`: This library provides the Selenium WebDriver API.
- `selenium.webdriver.remote.webelement`: This library provides the WebElement class for interacting with web elements.
- `inspect`: This library provides functions for inspecting code objects.
- `os`: This library provides functions for interacting with the operating system.
- `string`: This library provides functions for working with strings.
- `selenium.webdriver.common.keys`: This library provides the Keys class for simulating keyboard events.
- `selenium.webdriver.support.ui`: This library provides the WebDriverWait class for waiting for elements to be present.
- `typing`: This library provides type hints for function parameters and return values.
- `selenium.webdriver.support`: This library provides the expected_conditions class for waiting for elements to be present.
- `selenium.webdriver.common.by`: This library provides the By class for locating elements.
- `types`: This library provides functions for working with types.
- `selenium.webdriver.common.action_chains`: This library provides the ActionChains class for performing actions on web elements.

## Classes

### SeleniumTools
The SeleniumTools class provides various methods for interacting with Selenium WebDrivers.

#### Methods
- `__init__`: Initializes the SeleniumTools instance with a WebDriver instance and optional parameters.
- `__exit__`: Releases the resources occupied with the current session.
- `__enter__`: Returns an Selenium WebDriver instance.
- `_load_driver`: Creates a Selenium WebDriver instance.
- `_close`: Closes the Selenium WebDriver instance.
- `_attach_custom_parsers`: Attaches custom parsers from the given parser class to the LxmlParser class.
- `parse`: Parses an HTML element using the specified tag and locator.
- `_get_supported_browsers`: Returns a list of all supported browsers by Selenium.
- `_validate_driver`: Validates the Selenium WebDriver instance.
- `sessionid`: Returns the session ID of the WebDriver instance.
- `_is_valid_html`: Modifies the provided string based on its type.
- `get`: Visits the given URL or local HTML file or html content using the Selenium WebDriver instance.
- `get_locator`: Returns a locator tuple for the specified attribute value and locator type.
- `click`: Clicks on an element with the specified locator within the given Selenium WebDriver instance.
- `get_element`: Returns an element or a list of elements using the specified locator type and text.
- `select_option`: Selects a dropdown option based on the specified criteria.
- `fill`: Inserts or selects values using the specified criteria for a collection of form elements.
- `press_multiple_keys`: Presses multiple keys simultaneously using Selenium.
- `cookies`: Returns the cookies of the given Selenium WebDriver instance as a dictionary.
- `set_cookies`: Adds cookies to the given Selenium WebDriver instance.
- `execute_js`: Execute a JavaScript statement using the given Selenium WebDriver instance and returns the result.
- `text`: Returns the HTML source code of the currently loaded page in the given Selenium WebDriver instance.
- `url`: Returns the current loaded URL in the given Selenium WebDriver instance.
- `wait_for_element`: Waits for an element to be present and visible on the page.
- `element_visibility`: Toggles the visibility of an element on the page.

## Functions

### None

## Usage Example

```python
from seleniumtools import SeleniumTools

# Create a SeleniumTools instance
selenium_tools = SeleniumTools(driver)

# Get the current URL
url = selenium_tools.url()
print("Current URL:", url)

# Click on an element with the specified locator
selenium_tools.click("locator_text", "id")

# Get an element using the specified locator
element = selenium_tools.get_element("locator_text", "id")
print("Element:", element)

# Select a dropdown option
selenium_tools.select_option(element, "option_value", 0)
```

Note: The above code snippet is a simplified example and may not cover all the functionality of the SeleniumTools class.