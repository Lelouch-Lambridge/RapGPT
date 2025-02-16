import os
import requests
import json
import time
import re
from bs4 import BeautifulSoup

# 🔑 Replace with your Genius API token
GENIUS_API_TOKEN = "Yw7wCnsCutpmDlIjkpXf_6rVDuy9X2ymk0zB6xo8HUxJqaNxW-WGz6IVwbuDVa2J"

# Headers for API requests
HEADERS = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}

# Base URL for API
GENIUS_API_URL = "https://api.genius.com"

# List of words to exclude different versions of the same song
EXCLUDED_KEYWORDS = [
  "Live at", "Remix", "Mixed", "Version", "Alternate", "Demo", "Re-Recorded", "Acoustic", "Duplicate"
]


def get_artist_id(artist_name):
  """Fetch the Genius artist ID, handling redirections."""
  search_url = f"{GENIUS_API_URL}/search"
  params = {"q": artist_name}
  response = requests.get(search_url, headers=HEADERS, params=params).json()

  for hit in response["response"]["hits"]:
    primary_artist = hit["result"]["primary_artist"]["name"]
    redirected_artist_name = primary_artist
    print(f"Redirected to: {redirected_artist_name}")
    return hit["result"]["primary_artist"]["id"], redirected_artist_name
  
  return None, None


def get_artist_songs(artist_id, artist_name, include_features=True):
  """
  Generator that yields song URLs one by one.
  If include_features=False, it excludes songs where the artist is only featured.
  """
  seen_titles = set()
  page = 1

  while True:  # Loop to fetch all available songs page by page
    url = f"{GENIUS_API_URL}/artists/{artist_id}/songs"
    params = {"per_page": 50, "page": page}
    response = requests.get(url, headers=HEADERS, params=params).json()

    if "response" not in response or not response["response"]["songs"]: break  # Stop when there are no more songs

    for song in response["response"]["songs"]:
      song_title = song["title"]
      primary_artist = song["primary_artist"]["name"]

      clean_title = re.sub(r"\s*\(.*?\)", "", song_title).strip()

      if clean_title.lower() in seen_titles: continue

      if any(keyword.lower() in song_title.lower() for keyword in EXCLUDED_KEYWORDS): continue

      # If user chooses not to allow features, exclude songs where the artist is not the primary artist
      if not include_features and primary_artist.lower() != artist_name.lower(): continue

      seen_titles.add(clean_title.lower())
      yield {"title": song_title, "url": song["url"]}

    page += 1
    time.sleep(1)  # Avoid rate limiting


def scrape_lyrics(song_url, artist_name):
  """
  Scrape lyrics from a Genius song page.
  If the song has multiple artists, extract only the artist’s verses.
  If the song has no features, return the full lyrics.
  """
  session = requests.Session()
  response = session.get(song_url, headers=HEADERS, allow_redirects=True)  
  soup = BeautifulSoup(response.text, "html.parser")

  lyrics_divs = soup.find_all("div", class_="Lyrics-sc-37019ee2-1 jRTEBZ")
  
  if not lyrics_divs:
    print(f"❌ Lyrics not found for {song_url}")
    return None

  #Concatenate text from all found divs
  full_lyrics = "\n".join(div.get_text(separator="\n").strip() for div in lyrics_divs)

  #Check if there are **any** bracketed sections indicating a feature
  if not re.search(r"\[.*?:.*?\]", full_lyrics):
    print(f"✅ No features detected in {song_url}, returning full lyrics.")
    return full_lyrics  # Return full lyrics if the song is solo

  #Extract only artist’s verses if features exist
  verse_pattern = (
    rf"\[(?:Verse|Chorus|Bridge|Outro|Intro) \d*:?\s*{re.escape(artist_name)}\](.*?)\n(?=\[|\Z)"
    f"|"
    rf"\[{re.escape(artist_name)}(?:.*?)\](.*?)\n(?=\[|\Z)"
  )
  matches = re.findall(verse_pattern, full_lyrics, re.DOTALL)

  artist_verses = [match[0].strip() if match[0] else match[1].strip() for match in matches if any(match)]

  if artist_verses:
    return "\n\n".join(artist_verses)

  print(f"⚠️ No specific verses found for {artist_name} in {song_url}")
  return None


def save_lyrics_to_file(file_path, song_title, lyrics):
  """Append lyrics to a file incrementally to save memory."""
  with open(file_path, "a", encoding="utf-8") as f:
    f.write(f"\n\n--- {song_title} ---\n")
    f.write(lyrics + "\n")


def main(artist_name, max_songs=None, file_format="txt", include_features=True):
  """
  Main function to scrape an artist's lyrics.
  """
  print(f"Fetching Genius artist ID for '{artist_name}'...")
  artist_id, redirected_name = get_artist_id(artist_name)

  if not artist_id:
    print(f"Error: Could not find artist '{artist_name}' on Genius.")
    return

  file_path = f"lyrics/{redirected_name.replace(' ', '_').lower()}.{file_format}"
  os.makedirs("lyrics", exist_ok=True)

  print(f"Fetching {'all' if max_songs is None else max_songs} songs for '{redirected_name}', {'including' if include_features else 'excluding'} features...")

  song_count = 0
  for song in get_artist_songs(artist_id, redirected_name, include_features=include_features):
    if max_songs and song_count >= max_songs: break

    print(f"Scraping: {song['title']}")
    lyrics = scrape_lyrics(song["url"], redirected_name)
    if lyrics:
      save_lyrics_to_file(file_path, song["title"], lyrics)
      song_count += 1
    time.sleep(1)

  print(f"✅ Done! Lyrics for '{redirected_name}' scraped successfully and saved to {file_path}.")


if __name__ == "__main__":
  artist_name = input("Enter artist name: ")
  
  max_songs = input("Enter max number of songs to scrape (or type 'all' for all songs): ").strip().lower()
  max_songs = None if max_songs == "all" else int(max_songs)

  file_format = input("Save as (txt/json)? ").strip().lower() or "txt"
  
  include_features = input("Include featured songs? (yes/no): ").strip().lower()
  include_features = include_features in ["yes", "y"]

  main(artist_name, max_songs, file_format, include_features)
