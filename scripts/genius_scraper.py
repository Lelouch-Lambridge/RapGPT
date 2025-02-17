import os
import requests
import time
import re
import logging 
from itertools import cycle
import threading
import sys
import sqlite3
from bs4 import BeautifulSoup

with open("Genius_Api_Token", "r", encoding="utf-8") as token_file:
  GENIUS_API_TOKEN = token_file.read().strip()

HEADERS = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}

GENIUS_API_URL = "https://api.genius.com"

# List of words to exclude different versions of the same song
EXCLUDED_KEYWORDS = [
  "Live at", "Remix", "Mixed", "Version", "Alternate", "Demo", "Re-Recorded", "Acoustic", "Duplicate"
]

os.makedirs("logs", exist_ok=True)

info_logger = logging.getLogger("info_logger")
info_logger.setLevel(logging.INFO)
info_handler = logging.FileHandler("logs/info.log", mode="a", encoding="utf-8")
info_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
info_logger.addHandler(info_handler)

error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.WARNING)
error_handler = logging.FileHandler("logs/error.log", mode="a", encoding="utf-8")
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
error_logger.addHandler(error_handler)


SPINNER_FRAMES = cycle(["⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇"])
SPINNER_RUNNING = True

DB_FILE = "datasets.db"

def get_db_connection():
  """Returns a database connection."""
  return sqlite3.connect(DB_FILE)

def init_artist_table(artist):
  """Creates a new table for each artist (ensuring valid SQLite table names)."""
  table_name = artist.replace(" ", "_").lower()
  
  if table_name[0].isdigit(): table_name = f"artist_{table_name}"

  table_name = re.sub(r"[^a-zA-Z0-9_]", "", table_name)  
  
  conn = get_db_connection()
  cursor = conn.cursor()
  
  cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      song_title TEXT NOT NULL UNIQUE,
      lyrics TEXT NOT NULL
    )
  """)
  
  conn.commit()
  conn.close()

  return table_name

def save_lyrics_to_db(artist, song_title, lyrics):
  """Insert lyrics into the SQLite database under the artist-specific table."""
  table_name = init_artist_table(artist)

  conn = get_db_connection()
  cursor = conn.cursor()
  
  try:
    cursor.execute(f'INSERT INTO "{table_name}" (song_title, lyrics) VALUES (?, ?)', (song_title, lyrics))
    conn.commit()
    info_logger.info(f"✅ Saved: {song_title} by {artist} to database.")
  except sqlite3.IntegrityError: error_logger.warning(f"⚠️ Skipping duplicate: {song_title} by {artist}.")
  finally: conn.close()

def spinner():
  """Display a rotating spinner while scraping lyrics."""
  while SPINNER_RUNNING:
    sys.stdout.write(f"\rScraping... {next(SPINNER_FRAMES)} ")
    sys.stdout.flush()
    time.sleep(0.1)

def get_artist_id(artist_name):
  """Fetch the Genius artist ID, handling redirections."""
  search_url = f"{GENIUS_API_URL}/search"
  params = {"q": artist_name}
  response = requests.get(search_url, headers=HEADERS, params=params).json()

  for hit in response["response"]["hits"]:
    primary_artist = hit["result"]["primary_artist"]["name"]
    redirected_artist_name = primary_artist
    info_logger.info(f"Redirected to: {redirected_artist_name}")
    return hit["result"]["primary_artist"]["id"], redirected_artist_name
  
  error_logger.error(f"Artist '{artist_name}' not found on Genius.")
  return None, None

def get_artist_songs(artist_id, artist_name, include_features=True):
  """Generator that yields song URLs one by one."""
  seen_titles = set()
  page = 1

  while True:
    url = f"{GENIUS_API_URL}/artists/{artist_id}/songs"
    params = {"per_page": 50, "page": page}
    response = requests.get(url, headers=HEADERS, params=params).json()

    if "response" not in response or not response["response"]["songs"]:
      break  

    for song in response["response"]["songs"]:
      song_title = song["title"]
      primary_artist = song["primary_artist"]["name"]

      clean_title = re.sub(r"\s*\(.*?\)", "", song_title).strip()

      if clean_title.lower() in seen_titles: continue
      if any(keyword.lower() in song_title.lower() for keyword in EXCLUDED_KEYWORDS): continue
      if not include_features and primary_artist.lower() != artist_name.lower(): continue

      seen_titles.add(clean_title.lower())
      yield {"title": song_title, "url": song["url"]}

    page += 1
    time.sleep(1)

def scrape_lyrics(song_url, artist_name):
  """Scrape lyrics from Genius and extract only the artist’s verses if featured."""
  session = requests.Session()
  response = session.get(song_url, headers=HEADERS, allow_redirects=True)
  soup = BeautifulSoup(response.text, "html.parser")

  lyrics_divs = soup.find_all("div", class_="Lyrics-sc-37019ee2-1 jRTEBZ")

  if not lyrics_divs:
    error_logger.error(f"Lyrics not found for {song_url}")
    return None

  full_lyrics = "\n".join(div.get_text(separator="\n").strip() for div in lyrics_divs)

  if not re.search(r"\[.*?:.*?\]", full_lyrics):
    info_logger.info(f"No features detected in {song_url}, returning full lyrics.")
    return full_lyrics  

  verse_pattern = (
    rf"\[(?:Verse|Chorus|Bridge|Outro|Intro) \d*:?\s*{re.escape(artist_name)}\](.*?)\n(?=\[|\Z)"
    f"|"
    rf"\[{re.escape(artist_name)}(?:.*?)\](.*?)\n(?=\[|\Z)"
  )
  matches = re.findall(verse_pattern, full_lyrics, re.DOTALL)

  artist_verses = [match[0].strip() if match[0] else match[1].strip() for match in matches if any(match)]

  if artist_verses: return "\n\n".join(artist_verses)

  error_logger.warning(f"No specific verses found for {artist_name} in {song_url}")
  return None

def main(artist_name, max_songs=None, include_features=True):
  """Main function to scrape and store lyrics."""
  info_logger.info(f"Fetching Genius artist ID for '{artist_name}'...")
  artist_id, redirected_name = get_artist_id(artist_name)

  if not artist_id: return

  init_artist_table(redirected_name)

  global SPINNER_RUNNING
  SPINNER_RUNNING = True
  spinner_thread = threading.Thread(target=spinner, daemon=True)
  spinner_thread.start()

  song_count = 0
  for song in get_artist_songs(artist_id, redirected_name, include_features=include_features):
    if max_songs and song_count >= max_songs: break  

    info_logger.info(f"Scraping: {song['title']}")
    lyrics = scrape_lyrics(song["url"], redirected_name)
    if lyrics:
      save_lyrics_to_db(redirected_name, song["title"], lyrics)
      song_count += 1
    time.sleep(1)

  SPINNER_RUNNING = False
  spinner_thread.join()
  
  sys.stdout.write("\r✅ Done! Lyrics scraped successfully.    \n")
  sys.stdout.flush()

  info_logger.info(f"✅ Lyrics for '{redirected_name}' saved to database.")

if __name__ == "__main__":
  artist_name = input("Enter artist name: ")
  max_songs = input("Enter max number of songs (or 'all'): ").strip().lower()
  max_songs = None if max_songs == "all" else int(max_songs)

  include_features = input("Include featured songs? (yes/no): ").strip().lower() in ["yes", "y"]

  main(artist_name, max_songs, include_features)
