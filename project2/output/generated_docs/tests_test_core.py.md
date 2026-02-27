# test_core.py

## Overview
This module contains unit tests for the SeleniumTools class, which provides a set of tools for automating web browsers using Selenium WebDriver. The tests cover various aspects of the SeleniumTools class, including getting supported browsers, session ID, URLs, HTML content, parsing, executing JavaScript, getting locators, clicking elements, getting elements, pressing multiple keys, retrieving and setting cookies, waiting for elements, and checking element visibility.

## Dependencies
### Internal Dependencies
- `s_tool.core`: This module is imported internally to access the SeleniumTools class.

### External Libraries Used
- `webdriver_manager.chrome`: Used to manage ChromeDriver.
- `unittest`: Used for unit testing.
- `lxml.html`: Used for parsing HTML.
- `os`: Used for interacting with the operating system.
- `selenium.webdriver.chrome.options`: Used to configure ChromeDriver options.
- `selenium.webdriver.common.by`: Used to specify the locator strategy.
- `selenium.webdriver.remote.webelement`: Used to interact with web elements.
- `selenium`: Used for automating web browsers.
- `selenium.webdriver.chrome.service`: Used to manage ChromeDriver services.

## Classes

### LXMLParser
This class provides a method for parsing HTML links using the lxml library.

#### Methods
- `link(self, html_string, **kwargs)`: This method takes an HTML string and returns a list of links.

### SeleniumToolsTestCase
This class contains unit tests for the SeleniumTools class.

#### Methods
- `setUpClass(cls)`: This method sets up the test class by creating a ChromeDriver instance and initializing the SeleniumTools instance.
- `tearDownClass(cls)`: This method tears down the test class by quitting the ChromeDriver instance.
- `test_get_supported_browsers(self)`: This method tests the _get_supported_browsers method of the SeleniumTools class.
- `test_sessionid(self)`: This method tests the sessionid method of the SeleniumTools class.
- `test_get(self)`: This method tests the get method of the SeleniumTools class.
- `test_get_local_file(self)`: This method tests the get method of the SeleniumTools class with a local file.
- `test_get_html_content(self)`: This method tests the get method of the SeleniumTools class with HTML content.
- `test_parse(self)`: This method tests the parse method of the SeleniumTools class.
- `test_url(self)`: This method tests the url method of the SeleniumTools class.
- `test_text(self)`: This method tests the text method of the SeleniumTools class.
- `test_execute_js(self)`: This method tests the execute_js method of the SeleniumTools class.
- `test_get_locator(self)`: This method tests the get_locator method of the SeleniumTools class.
- `test_click(self)`: This method tests the click method of the SeleniumTools class.
- `test_get_element(self)`: This method tests the get_element method of the SeleniumTools class.
- `test_press_multiple_keys(self)`: This method tests the press_multiple_keys method of the SeleniumTools class.
- `test_cookies(self)`: This method tests the cookies method of the SeleniumTools class.
- `test_set_cookies(self)`: This method tests the set_cookies method of the SeleniumTools class.
- `test_wait_for_element(self)`: This method tests the wait_for_element method of the SeleniumTools class.
- `test_element_visibility(self)`: This method tests the element_visibility method of the SeleniumTools class.

## Functions
None

## Usage Example
```python
import unittest
from s_tool.core import SeleniumTools

class TestSeleniumTools(unittest.TestCase):
    def test_selenium_tools(self):
        selenium_tools = SeleniumTools()
        selenium_tools.get("https://www.example.com")
        self.assertEqual(selenium_tools.url(), "https://www.example.com")

if __name__ == "__main__":
    unittest.main()
```