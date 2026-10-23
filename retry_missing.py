import requests

for i in range(1, 1026):
    try:
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}", timeout=10)
        if r.status_code != 200:
            continue

        name = r.json()["name"].capitalize()

        import os
        if not os.path.exists(f"dataset/{name}"):
            print(i, name)

    except Exception:
        pass