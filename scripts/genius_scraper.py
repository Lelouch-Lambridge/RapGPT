import requests
import time
import re
import sqlite3
import multiprocessing
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
from spinner import Spinner
from logger import Logger

# Configuration
MODEL_NAME = "models/stablelm_base"
DB_FILE = "lyrics.db"
TOKENIZED_DB_FILE = "tokenized_lyrics.db"
GENIUS_API_URL = "https://api.genius.com"
EXCLUDED_KEYWORDS = ["Live at", "Remix", "Mixed", "Version", "Alternate", "Demo", "Re-Recorded", "Acoustic", "Duplicate"]

with open("Genius_Api_Token", "r", encoding="utf-8") as token_file:
  GENIUS_API_TOKEN = token_file.read().strip()
  HEADERS = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}

logger = Logger()

def load_tokenizer():
  global tokenizer
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer = None
load_tokenizer()

def tokenize_lyrics(lyrics, queue):
  tokenized_data = tokenizer([lyrics], truncation=True, padding="max_length", max_length=256)
  queue.put(tokenized_data)

def get_db_connection(db_file):
  return sqlite3.connect(db_file)

def get_artist_id(artist_name):
  response = requests.get(f"{GENIUS_API_URL}/search", headers=HEADERS, params={"q": artist_name}).json()
  for hit in response.get("response", {}).get("hits", []):
    return hit["result"]["primary_artist"]["id"], hit["result"]["primary_artist"]["name"]
  logger.error(f"Artist '{artist_name}' not found on Genius.")
  return None, None

def get_artist_songs(artist_id):
  page, seen_titles = 1, set()
  while True:
    response = requests.get(f"{GENIUS_API_URL}/artists/{artist_id}/songs", headers=HEADERS, params={"per_page": 50, "page": page}).json()
    if "response" not in response or not response["response"].get("songs"): break
    
    for song in response["response"]["songs"]:
      clean_title = re.sub(r"\s*\(.*?\)", "", song["title"]).strip()
      if clean_title.lower() in seen_titles or any(k.lower() in song["title"].lower() for k in EXCLUDED_KEYWORDS): continue
      
      seen_titles.add(clean_title.lower())
      yield song["title"], song["url"]
    page += 1
    time.sleep(1)

def init_artist_table(cursor, artist):
  table_name = re.sub(r"[^a-zA-Z0-9_]", "", artist.replace(" ", "_").lower())
  if table_name[0].isdigit():
    table_name = f"artist_{table_name}"
  cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      song_title TEXT NOT NULL UNIQUE,
      lyrics TEXT NOT NULL
    )
  """)
  return table_name

def save_lyrics_and_tokenized(cursor_lyrics, cursor_tokens, artist, song_title, lyrics, tokenized_data):
  table_name = init_artist_table(cursor_lyrics, artist)
  try:
    cursor_lyrics.execute(f'INSERT INTO "{table_name}" (song_title, lyrics) VALUES (?, ?)', (song_title, lyrics))
  except sqlite3.IntegrityError:
    logger.warning(f"⚠️ Skipping duplicate: {song_title} by {artist}.")
    return
  token_table_name = table_name + "_tokenized"
  cursor_tokens.execute(f"""
    CREATE TABLE IF NOT EXISTS {token_table_name} (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      input_ids TEXT,
      attention_mask TEXT
    )
  """)
  input_ids_list = tokenized_data["input_ids"]
  attention_mask_list = tokenized_data["attention_mask"]
  data_to_insert = [(" ".join(map(str, ids)), " ".join(map(str, mask))) for ids, mask in zip(input_ids_list, attention_mask_list)]
  cursor_tokens.executemany(f"INSERT INTO {token_table_name} (input_ids, attention_mask) VALUES (?, ?)", data_to_insert)

def scrape_lyrics(song_url, artist_name):
  session = requests.Session()
  response = session.get(song_url, headers=HEADERS, allow_redirects=True)
  soup = BeautifulSoup(response.text, "html.parser")
  
  lyrics_divs = soup.find_all("div", class_="Lyrics-sc-37019ee2-1 jRTEBZ")
  if not lyrics_divs:
    logger.error(f"Lyrics not found for {song_url}")
    return None

  full_lyrics = "\n".join(div.get_text(separator="\n").strip() for div in lyrics_divs)
  
  if not re.search(r"\[.*?:.*?\]", full_lyrics):
    logger.info(f"No features detected in {song_url}, returning full lyrics.")
    return full_lyrics  

  verse_pattern = (
    rf"\[(?:Verse|Chorus|Bridge|Outro|Intro) \d*:?\s*{re.escape(artist_name)}\](.*?)\n(?=\[|\Z)"
    f"|"
    rf"\[{re.escape(artist_name)}(?:.*?)\](.*?)\n(?=\[|\Z)"
  )
  matches = re.findall(verse_pattern, full_lyrics, re.DOTALL)

  artist_verses = [match[0].strip() if match[0] else match[1].strip() for match in matches if any(match)]
  
  if artist_verses: return "\n\n".join(artist_verses)

  logger.warning(f"No specific verses found for {artist_name} in {song_url}")
  return None

def main(artist_name, max_songs=None):
  artist_id, redirected_name = get_artist_id(artist_name)
  if not artist_id: return
  conn_lyrics = get_db_connection(DB_FILE)
  conn_tokens = get_db_connection(TOKENIZED_DB_FILE)
  cursor_lyrics = conn_lyrics.cursor()
  cursor_tokens = conn_tokens.cursor()

  print(f"\rFetching Lyrics for {redirected_name}...")
  
  spinner = Spinner.get_instance("Scraping lyrics...")
  spinner.start()
  song_count = 0
  for song_title, song_url in get_artist_songs(artist_id):
    if max_songs and song_count >= max_songs: break
    
    logger.info(f"Scraping: {song_title}")
    lyrics = scrape_lyrics(song_url, redirected_name)
    if lyrics:
      queue = multiprocessing.Queue()
      tokenizing_process = multiprocessing.Process(target=tokenize_lyrics, args=(lyrics, queue))
      tokenizing_process.start()
      tokenizing_process.join()
      tokenized_data = queue.get()
      save_lyrics_and_tokenized(cursor_lyrics, cursor_tokens, redirected_name, song_title, lyrics, tokenized_data)
      song_count += 1
    
  spinner.stop()
  conn_lyrics.commit()
  conn_tokens.commit()
  conn_lyrics.close()
  conn_tokens.close()
  
  print(f"\r✅ Done! Lyrics for {song_count} songs scraped successfully.")
  logger.info(f"✅ Lyrics for '{redirected_name}' saved to database.")

if __name__ == "__main__":
  artist_name = input("Enter artist name: ")
  max_songs = input("Enter max number of songs (or 'all'): ").strip().lower()
  max_songs = None if max_songs == "all" else int(max_songs)
  main(artist_name, max_songs)
