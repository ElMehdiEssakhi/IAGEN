# run.py
import asyncio
from graph import construire_graphe

app = construire_graphe()

async def main():
    res = await app.ainvoke(
        {"question": "Comment mettre mon enceinte SoundGo X1 en mode appairage ?"},
        {"recursion_limit": 10},
    )
    print(res["categorie"], res["reponse_finale"], res.get("sources", []))

if __name__ == "__main__":
    asyncio.run(main())