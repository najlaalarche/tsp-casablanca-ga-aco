"""Client Gemini (SDK officiel google-genai) : choix du modèle, délais, repli, JSON, grounding.

La clé est lue dans .env (GEMINI_API_KEY) et n'est jamais écrite dans les journaux.
"""
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass

from config.settings import BASE_DIR

log = logging.getLogger(__name__)

# Ordre de préférence. Vérifié le 2026-11-16 via models.list() puis appels réels :
# - gemini-2.5-* : listés mais refusés (404 « no longer available to new users ») ;
# - gemini-3.8-flash / 3.7-flash : disponibles mais souvent saturés (503/504) ;
# - gemini-3.6-flash, 3.5-flash-lite, 3.1-flash-lite : répondent.
PREFERRED_MODELS = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
TIMEOUT_MS = 90_000          # génération complète
PROBE_TIMEOUT_MS = 20_000    # test de disponibilité d'un modèle
RETRIABLE = {429, 500, 503, 504}


class GeminiConfigError(RuntimeError):
    """Clé absente/invalide ou SDK manquant : message destiné à l'utilisateur."""


class GeminiCallError(RuntimeError):
    pass


class SearchUnavailable(GeminiCallError):
    """L'outil Google Search grounding est refusé (quota/palier) ou indisponible."""


@dataclass
class GeminiReply:
    text: str
    model: str
    sources: list[tuple[str, str]]  # (titre, uri)
    grounded: bool
    queries: list[str]


def load_api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env", override=False)
    except ImportError:
        pass
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise GeminiConfigError(
            "Clé Gemini absente.\n\nCréez le fichier .env à la racine du projet (copie de .env.example) "
            "et renseignez :\nGEMINI_API_KEY=votre_clé\n\nClé gratuite : https://aistudio.google.com/apikey")
    return key


def _code(e: Exception) -> int | None:
    c = getattr(e, "code", None)
    return c if isinstance(c, int) else None


def _short(e: Exception) -> str:
    msg = getattr(e, "message", None) or str(e)
    return msg.split("\n")[0][:200]


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise GeminiConfigError("SDK google-genai non installé (pip install google-genai).") from e
        self.types = types
        self.client = genai.Client(api_key=api_key or load_api_key(),
                                   http_options=types.HttpOptions(timeout=TIMEOUT_MS))
        forced = (model or os.getenv("GEMINI_MODEL", "")).strip()
        self.candidates = [forced] if forced else list(PREFERRED_MODELS)
        self.model: str | None = None  # modèle effectivement utilisé (premier qui répond)
        self.dead: set[str] = set()    # modèles en échec pendant cette session
        self.probe_log: list[str] = []

    # ------------------------------------------------------------ modèles
    def available_models(self) -> list[str]:
        try:
            return [m.name.removeprefix("models/") for m in self.client.models.list()
                    if "generateContent" in (m.supported_actions or [])]
        except Exception as e:  # noqa: BLE001
            code = _code(e)
            if code in (400, 401, 403):
                raise GeminiConfigError(f"Clé Gemini refusée ({code}) : {_short(e)}") from e
            raise GeminiCallError(f"Impossible de lister les modèles Gemini : {_short(e)}") from e

    def _order(self) -> list[str]:
        alive = [m for m in self.candidates if m not in self.dead] or self.candidates
        if self.model in alive:
            return [self.model] + [m for m in alive if m != self.model]
        return alive

    def select_model(self) -> str:
        """Choisit le premier modèle de la liste qui répond vite (appel minimal, délai court).

        Évite de perdre plusieurs minutes sur un modèle saturé lors des vrais appels.
        """
        t = self.types
        cfg = t.GenerateContentConfig(http_options=t.HttpOptions(timeout=PROBE_TIMEOUT_MS),
                                      max_output_tokens=5)
        for model in self.candidates:
            t0 = time.time()
            try:
                self.client.models.generate_content(model=model, contents="Réponds OK.", config=cfg)
                self.model = model
                self.probe_log.append(f"{model} : OK ({time.time() - t0:.1f} s)")
                return model
            except Exception as e:  # noqa: BLE001
                code = _code(e)
                if code in (400, 401, 403) and "API key" in _short(e):
                    raise GeminiConfigError(f"Clé Gemini invalide ({code}) : {_short(e)}") from e
                self.dead.add(model)
                self.probe_log.append(f"{model} : indisponible ({code or type(e).__name__})")
                log.warning("Modèle %s indisponible : %s", model, _short(e))
        raise GeminiCallError("Aucun modèle Gemini disponible :\n" + "\n".join(self.probe_log))

    # ------------------------------------------------------------ appels
    def generate(self, prompt: str, system: str, *, json_schema: dict | None = None,
                 use_search: bool = False, temperature: float = 0.4) -> GeminiReply:
        """Appelle le premier modèle disponible. use_search active Google Search grounding."""
        t = self.types
        cfg = dict(system_instruction=system, temperature=temperature)
        if use_search:
            cfg["tools"] = [t.Tool(google_search=t.GoogleSearch())]
        elif json_schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = json_schema
        config = t.GenerateContentConfig(**cfg)

        errors = []
        for model in self._order():
            for attempt in range(2):
                try:
                    r = self.client.models.generate_content(model=model, contents=prompt, config=config)
                    text = r.text or ""
                    if not text.strip():
                        raise GeminiCallError("réponse vide")
                    self.model = model
                    return self._reply(r, model, text)
                except GeminiCallError as e:
                    errors.append(f"{model}: {e}")
                    break
                except Exception as e:  # noqa: BLE001
                    code = _code(e)
                    if code in (400, 401, 403) and "API key" in _short(e):
                        raise GeminiConfigError(f"Clé Gemini invalide ({code}) : {_short(e)}") from e
                    if use_search and code == 429:
                        # Le 429 immédiat avec l'outil de recherche = quota grounding indisponible.
                        raise SearchUnavailable(
                            f"Google Search grounding refusé par l'API (429 quota) : {_short(e)}") from e
                    errors.append(f"{model}: {code} {_short(e)}")
                    if code in (404, 504) or (code == 503 and attempt == 1):
                        self.dead.add(model)
                    log.warning("Gemini %s tentative %d : %s %s", model, attempt + 1, code, _short(e))
                    if code in (503, 500) and attempt == 0:
                        time.sleep(2)
                        continue
                    break  # 404/429/504/... : modèle suivant
        raise GeminiCallError("Aucun modèle Gemini n'a répondu.\n" + "\n".join(errors[-6:]))

    @staticmethod
    def _reply(r, model: str, text: str) -> GeminiReply:
        sources, queries, grounded = [], [], False
        try:
            gm = r.candidates[0].grounding_metadata
        except (AttributeError, IndexError, TypeError):
            gm = None
        if gm is not None:
            for ch in gm.grounding_chunks or []:
                web = getattr(ch, "web", None)
                if web and web.uri:
                    sources.append((web.title or web.domain or web.uri, web.uri))
            queries = list(gm.web_search_queries or [])
            grounded = bool(sources or queries)
        return GeminiReply(text, model, sources, grounded, queries)


def to_json_text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)
