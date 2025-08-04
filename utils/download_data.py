import os
from dotenv import load_dotenv
import dropbox

load_dotenv()

ACCESS_TOKEN = os.environ.get('DROPBOX_ACCESS_TOKEN')
DROPBOX_FOLDER_PATH = os.environ.get('DROPBOX_FOLDER_PATH')
LOCAL_DATA_DIR = "./data/"

def download_folder(dbx, dropbox_path, local_path):
    os.makedirs(local_path, exist_ok=True)

    result = dbx.files_list_folder(dropbox_path)

    for entry in result.entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            local_file_path = os.path.join(local_path, entry.name)
            print(f"Downloading {entry.path_lower} to {local_file_path}...")
            _, res = dbx.files_download(entry.path_lower)
            with open(local_file_path, "wb") as f:
                f.write(res.content)

        elif isinstance(entry, dropbox.files.FolderMetadata):
            download_folder(dbx, entry.path_lower, os.path.join(local_path, entry.name))

def main():
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    download_folder(dbx, DROPBOX_FOLDER_PATH, LOCAL_DATA_DIR)
    print("Download fertig!")

if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv
import dropbox

load_dotenv()


if ACCESS_TOKEN is None:
    raise ValueError("DROPBOX_ACCESS_TOKEN environment variable not set!")

dbx = dropbox.Dropbox(ACCESS_TOKEN)

result = dbx.files_list_folder("")  # root

print("Inhalte im Root:")
for entry in result.entries:
    print(f"{entry.name} ({type(entry).__name__})")
