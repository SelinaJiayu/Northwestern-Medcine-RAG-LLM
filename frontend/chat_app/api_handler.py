import requests
import time

class APIHandler:
    def __init__(self, api_url, max_retries=3, backoff_factor=1):
        """
        Initialize the APIHandler with the base API URL, maximum retries, and backoff factor.
        
        :param api_url: str - Base URL of the API
        :param max_retries: int - Maximum number of retries in case of request failure
        :param backoff_factor: int - Backoff factor for exponential backoff (in seconds)
        """
        self.api_url = api_url
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _retry_request(self, method, endpoint, **kwargs):
        """
        Helper method to handle retries with exponential backoff.
        
        :param method: str - HTTP method to be used ('GET' or 'POST')
        :param endpoint: str - API endpoint to send the request to
        :param kwargs: Additional arguments to be passed to the request
        :return: dict - JSON response from the API, or None if all retries fail
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                if endpoint is not None:
                    api_url = f"{self.api_url}/{endpoint}"
                else:
                    api_url = self.api_url
                if method == "GET":
                    response = requests.get(api_url, **kwargs)
                elif method == "POST":
                    response = requests.post(api_url, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** (attempt - 1))
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("All retry attempts failed.")
                    return None

    def get(self, endpoint=None, params=None):
        """
        Send a GET request to the specified endpoint with optional parameters and retry if it fails.
        
        :param endpoint: str - The API endpoint to send the GET request to
        :param params: dict - Optional dictionary of query parameters
        :return: dict - JSON response from the API, or None if all retries fail
        """
        return self._retry_request("GET", endpoint, params=params)

    def post(self, endpoint=None, data=None):
        """
        Send a POST request to the specified endpoint with optional JSON data and retry if it fails.
        
        :param endpoint: str - The API endpoint to send the POST request to
        :param data: dict - Optional dictionary of data to send in the request body
        :return: dict - JSON response from the API, or None if all retries fail
        """
        return self._retry_request("POST", endpoint, json=data)
