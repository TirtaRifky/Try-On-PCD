import requests
import os
import math

# --- Configuration ---
# 1. PASTE YOUR PEXELS API KEY HERE
API_KEY = "YOUR_API_KEY_GOES_HERE" 

# 2. Set your search and download parameters
SEARCH_QUERY = "animals"
IMAGES_TO_DOWNLOAD = 1000
DOWNLOAD_DIR = "non_faces"  # Using the existing non_faces folder
# --- End of Configuration ---

# Pexels API constants
API_URL = "https://api.pexels.com/v1/search"
IMAGES_PER_PAGE = 80  # This is the maximum allowed by Pexels

def setup_download_dir():
    """Ensures the download directory exists."""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        print(f"Created directory: {DOWNLOAD_DIR}")

def download_image(url, filepath):
    """Downloads a single image from a URL and saves it to a file."""
    # Check if file already exists
    if os.path.exists(filepath):
        print(f"Skipping {filepath} - already exists")
        return True
        
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded: {filepath}")
            return True
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

def fetch_pexel_images():
    """Fetches and downloads the specified number of images."""
    
    if API_KEY == "YOUR_API_KEY_GOES_HERE":
        print("Error: Please replace 'YOUR_API_KEY_GOES_HERE' with your actual Pexels API key.")
        return

    setup_download_dir()
    
    headers = {
        "Authorization": API_KEY
    }
    
    # Calculate the total number of pages needed
    total_pages = math.ceil(IMAGES_TO_DOWNLOAD / IMAGES_PER_PAGE)
    image_count = 0
    successful_downloads = 0

    print(f"Starting download of {IMAGES_TO_DOWNLOAD} images for query '{SEARCH_QUERY}'...")

    for page in range(1, total_pages + 1):
        if successful_downloads >= IMAGES_TO_DOWNLOAD:
            break

        params = {
            "query": SEARCH_QUERY,
            "per_page": IMAGES_PER_PAGE,
            "page": page
        }

        try:
            # Make the API request
            response = requests.get(API_URL, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                photos = data.get('photos', [])
                
                if not photos:
                    print("No more photos found. Stopping.")
                    break

                for photo in photos:
                    if successful_downloads >= IMAGES_TO_DOWNLOAD:
                        break

                    # Get image URL and create a unique filename
                    image_url = photo['src']['original']
                    image_id = photo['id']
                    file_extension = os.path.splitext(image_url.split('?')[0])[-1] or '.jpg'
                    filename = f"{image_id}{file_extension}"
                    filepath = os.path.join(DOWNLOAD_DIR, filename)
                    
                    # Download the image and track success
                    if download_image(image_url, filepath):
                        successful_downloads += 1
                    image_count += 1
            
            elif response.status_code == 429:
                print("Rate limit exceeded. Please wait and try again later.")
                break
            else:
                print(f"Error fetching page {page}. Status: {response.status_code}")
                print(f"Response: {response.text}")
                break

        except requests.exceptions.RequestException as e:
            print(f"A network error occurred: {e}")
            break

    print(f"\nDownload complete. Total images downloaded: {image_count}")

# --- Run the script ---
if __name__ == "__main__":
    fetch_pexel_images()