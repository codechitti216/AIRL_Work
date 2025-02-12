import requests
import xml.etree.ElementTree as ET

# Define the Hacker News RSS feed URL
hn_rss_url = "https://news.ycombinator.com/rss"

# Send a request to get the RSS feed
response = requests.get(hn_rss_url)
response.raise_for_status()

# Parse the XML content of the feed
root = ET.fromstring(response.content)

# Loop through each <item> in the feed
for item in root.findall(".//item"):
    title = item.find("title").text
    link = item.find("link").text

    # Check if '2FA' is mentioned in the title
    if '2FA' in title:
        # Manually checking comments, votes, or other attributes
        print("Link to post mentioning 2FA:", link)
        break
