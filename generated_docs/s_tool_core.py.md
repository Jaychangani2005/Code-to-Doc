# SeleniumTools Core Module Documentation

## Summary

This module provides a comprehensive utility class, `SeleniumTools`, for automating web browser interactions using Selenium WebDriver. It simplifies common tasks such as driver management, element locating, parsing HTML content, and navigating web pages, offering a streamlined interface for web scraping and testing.

## Architecture

The `SeleniumTools` class acts as a central orchestrator for Selenium operations. It initializes and manages a `SeleniumDriver` instance for browser automation and utilizes a `LxmlParser` for processing HTML content extracted from web pages. It also supports the integration of custom parsers, allowing for flexible data extraction logic. The class handles exceptions from Selenium and its own custom exceptions to provide robust error management.

## Functions

*   **`__init__(self, driver=None, **kwargs)`**: Initializes `SeleniumTools` with an optional WebDriver instance and configuration for browser, headless mode, executable path, and custom parsers.
*   **`__exit__(self, typesa, value, tracebacks)`**: Cleans up resources by closing the WebDriver session.
*   **`__enter__(self)`**: Returns the `SeleniumTools` instance, ensuring a WebDriver is available for use within a `with` statement.
*   **`_load_driver(self)`**: Creates and returns a new Selenium WebDriver instance based on the provided configuration.
*   **`_close(self)`**: Closes the current WebDriver session.
*   **`_attach_custom_parsers(self, parser_class: Type)`**: Dynamically adds methods from a custom parser class to the `LxmlParser`.
*   **`parse(self, ele_tag: str, locator_text: str, locator_type: str = "id", **kwargs)`**: Parses a specific HTML element using its tag and locator, leveraging the attached parser.
*   **`_get_supported_browsers(self)`**: Returns a list of browsers supported by the Selenium WebDriver.
*   **`_validate_driver(self)`**: Verifies if the provided or loaded WebDriver is valid and functional.
*   **`sessionid(self)`**: Retrieves the unique session ID of the WebDriver instance.
*   **`_is_valid_html(self, content: str)`**: Preprocesses input content to determine if it's a local file path, URL, or raw HTML, formatting it appropriately for the `get` method.
*   **`get(self, url_or_html: str)`**: Navigates the browser to a given URL, local HTML file, or directly to provided HTML content.
*   **`get_locator(self, locator_text: str, locator_type: str = "id")`**: Generates a Selenium locator tuple based on the provided text and type.