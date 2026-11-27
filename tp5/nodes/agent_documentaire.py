# tp5/nodes/agent_documentaire.py
import json
#from functools import lru_cache
from typing import Literal

from langchain.agents import create_agent
from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from llm import get_llm
from state import EtatSAV


PROMPT =(
            "Tu es l'agent de depannage d'un SAV. Utilise l'outil rag_multimodal "
            "pour chercher la reponse dans les manuels, puis reponds de facon claire "
            "en t'appuyant uniquement sur le resultat de l'outil. "
            "Si l'outil ne trouve rien de pertinent, dis-le sans inventer."
        )


client = MultiServerMCPClient({
    "manuels": {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable_http"}
})
_react = None

async def _get_react():
    global _react
    if _react is None:
        tools = await client.get_tools()
        _react = create_agent(model=get_llm(), tools=tools, system_prompt=PROMPT)
    return _react


async def agent_documentaire(etat: EtatSAV) -> Command[Literal["orchestrateur"]]:
    msg = etat["question"]
    if etat.get("image"):
        msg += f"\n(Photo de l'ecran jointe : {etat['image']})"

    react = await _get_react()
    resultat =await react.ainvoke({"messages": [{"role": "user", "content": msg}]})
    messages = resultat["messages"]

    # Récupère les sources depuis le retour des outils (ToolMessage).
    # Le serveur MCP renvoie une liste de blocs {"type": "text", "text": "<json>"} :
    # on aplatit les blocs, puis on fusionne les sources de chaque appel.
    sources: list[str] = []
    for m in messages:
        if m.type != "tool":
            continue
        contenu = m.content
        if isinstance(contenu, list):
            contenu = "".join(
                bloc.get("text", "") for bloc in contenu if isinstance(bloc, dict)
            )
        try:
            data = json.loads(contenu)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            sources.extend(data.get("sources", []))
    sources = list(dict.fromkeys(sources))  # sans doublons

    return Command(
        update={"reponse_finale": messages[-1].content, "sources": sources},
        goto="orchestrateur",
    )