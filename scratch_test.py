import requests

headers = {
    "User-Agent": "Sri Vyshnavi Konda srivyshnavi633@gmail.com"
}

url = "https://data.sec.gov/submissions/CIK0000320193.json"

response = requests.get(url, headers=headers)

print(response.status_code)

data = response.json()

print(data["name"])

print(data["filings"]["recent"]["form"][:10])
print(data["tickers"])