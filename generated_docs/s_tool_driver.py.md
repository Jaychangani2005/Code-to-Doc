# `s_tool.driver` Module

This module provides a flexible way to instantiate and manage Selenium WebDriver instances for various browsers. It leverages `webdriver-manager` to automatically download and configure the necessary browser drivers, simplifying the setup process for automated browser testing.

## Architecture

This module primarily interacts with the `selenium` library to create WebDriver instances. It uses `webdriver-manager` to handle the installation and path management of browser-specific drivers (Chrome, Firefox, and Internet Explorer). The `SeleniumDriver` class acts as a facade, abstracting the complexities of driver initialization and configuration.

## Functions

*   **`__init__(self, browser=None, headless=False, executable_path=None)`**: Initializes the `SeleniumDriver` with the desired browser, headless mode preference, and an optional executable path for the driver.
*   **`load_driver(self)`**: Creates and returns a WebDriver instance based on the configured browser. Raises a `ValueError` for unsupported browsers.
*   **`get_chrome_driver(self)`**: Returns an instance of `webdriver.Chrome`. It uses `ChromeDriverManager` to install the driver if not already present and configures Chrome options, including headless mode.
*   **`get_firefox_driver(self)`**: Returns an instance of `webdriver.Firefox`. It uses `GeckoDriverManager` for Firefox driver management and applies Firefox options, including headless mode.
*   **`get_ie_driver(self)`**: Returns an instance of `webdriver.Ie`. It uses `IEDriverManager` for Internet Explorer driver management and sets up IE options.
*   **`_get_chrome_options(self)`**: Helper method to create and configure `webdriver.ChromeOptions`, setting the `headless` attribute.
*   **`_get_firefox_options(self)`**: Helper method to create and configure `webdriver.FirefoxOptions`, setting the `headless` attribute.
*   **`_get_ie_options(self)`**: Helper method to create and configure `webdriver.IeOptions`.