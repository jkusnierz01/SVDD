import pandas as pd
import yt_dlp
import os
import json

def main():
    df = pd.read_csv("singfake.csv", sep=',')
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)
    
    log_list = []  
    
    for index, row in df.iterrows():
        url = row["Url"]
        singer = row["Singer"]
        title = row["Title"]
        spoof_type = row["Bonafide Or Spoof"]
        
        filename = f"{singer}_{title.replace('/', '_').replace(' ', '_')}_{spoof_type}.flac"
        filepath = os.path.join(output_dir, filename)
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'flac',
            'outtmpl': filepath,
            'noplaylist': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Pobrano: {filename}")
            log_list.append({
                "url": url,
                "filename": filename,
                "filepath": filepath,
                "status": "success"
            })
        except Exception as e:
            print(f"Błąd przy pobieraniu {url}: {e}")
            log_list.append({
                "url": url,
                "filename": filename,
                "filepath": filepath,
                "status": "error",
                "error_msg": str(e)
            })
    
    with open("download_log.json", "w", encoding="utf-8") as f:
        json.dump(log_list, f, indent=4, ensure_ascii=False)
    print("Log zapisany do download_log.json")

if __name__ == '__main__':
    main()