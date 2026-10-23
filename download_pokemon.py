import os
import requests
import time

SAVE_DIR = "dataset"
MAX_POKEMON = 1025

os.makedirs(SAVE_DIR, exist_ok=True)

for pokemon_id in range(1, MAX_POKEMON + 1):
    try:
        api_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"

        response = requests.get(api_url)

        if response.status_code != 200:
            print(f"Skipping #{pokemon_id}")
            continue

        data = response.json()

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

        print(f"{pokemon_id:04d} {name:<15} -> {count} images")

        time.sleep(0.1)

    except Exception as e:
        print(f"Error with #{pokemon_id}: {e}")

print("\nFinished downloading Pokémon dataset!")