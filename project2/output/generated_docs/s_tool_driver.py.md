# s_tool\driver.py

## Overview
The `s_tool\driver.py` module provides a class `SeleniumDriver` to create and manage instances of Selenium web drivers for various browsers. It allows users to select the browser, enable headless mode, and specify the executable path for the driver. The module uses the `webdriver_manager` library to automatically download and manage the browser drivers.

## Dependencies
### Internal Dependencies
None

### External Libraries Used
- `webdriver_manager.chrome`: For managing Chrome browser drivers.
- `webdriver_manager.firefox`: For managing Firefox browser drivers.
- `webdriver_manager.microsoft`: For managing Internet Explorer browser drivers.
- `selenium`: For creating and managing Selenium web drivers.
- `selenium.webdriver.chrome.service`: For creating Chrome driver services.
- `selenium.webdriver.ie.service`: For creating Internet Explorer driver services.

## Classes
### SeleniumDriver
The `SeleniumDriver` class is the main class in this module. It provides methods to create and manage instances of Selenium web drivers.

#### `__init__(self, browser=None, headless=False, executable_path=None)`
Initializes the `SeleniumDriver` instance with the specified browser, headless mode, and executable path.

#### `load_driver(self)`
Creates and returns a Selenium web driver instance based on the specified browser.

#### `get_chrome_driver(self)`
Returns a Chrome driver instance.

#### `get_firefox_driver(self)`
Returns a Firefox driver instance.

#### `get_ie_driver(self)`
Returns an Internet Explorer driver instance.

#### `_get_chrome_options(self)`
Returns the Chrome driver options with headless mode enabled.

#### `_get_firefox_options(self)`
Returns the Firefox driver options with headless mode enabled.

#### `_get_ie_options(self)`
Returns the Internet Explorer driver options.

## Functions
None

## Usage Example
```python
from s_tool.driver import SeleniumDriver

# Create a SeleniumDriver instance for Chrome browser
driver = SeleniumDriver(browser='chrome', headless=True)

# Load the Chrome driver instance
chrome_driver = driver.load_driver()

# Use the Chrome driver instance to navigate to a webpage
chrome_driver.get('https://www.example.com')
```
Note: Replace `https://www.example.com` with the actual URL you want to navigate to.