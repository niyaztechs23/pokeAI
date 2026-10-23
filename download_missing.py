import os
import requests

SAVE_DIR = "dataset"

for pokemon_id in [51, 52, 53]:
    api_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    data = requests.get(api_url).json()

    name = data["name"].capitalize()
    folder = os.path.join(SAVE_DIR, name)
    os.makedirs(folder, exist_ok=True)

    images = {
        "official.png": data["sprites"]["other"]["official-artwork"]["front_default"],
        "home.png": data["sprites"]["other"]["home"]["front_default"],
        "front.png": data["sprites"]["front_default"],
        "front_shiny.png": data["sprites"]["front_shiny"],
        "back.png": data["sprites"]["back_default"],
        "back_shiny.png": data["sprites"]["back_shiny"],
    }

    count = 0
    for filename, url in images.items():
        if url:
            img = requests.get(url)
            with open(os.path.join(folder, filename), "wb") as f:
                f.write(img.content)
            count += 1

    print(f"{pokemon_id:04d} {name} -> {count} images")

print("Done!")