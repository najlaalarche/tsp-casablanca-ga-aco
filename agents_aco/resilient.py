"""Client Gemini « résilient » pour le SMA ACO (l'application AG garde son propre client, non modifié).

- 503 (ou 500, surcharge serveur) : nouvelles tentatives avec attente progressive 2 s, 5 s, 10 s ;
- puis bascule sur les autres modèles qui répondaient au démarrage de l'application ;
- budget de 60 s par agent (attentes et appels compris) avant de déclarer l'échec.

La fonction d'appel est injectable : les tests utilisent un faux appel qui simule le 503, sans réseau.
"""
from __future__ import annotations
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable

from agents.gemini_client import GeminiCallError, GeminiConfigError, GeminiReply, _code, _short
from config.settings import CACHE_DIR

log = logging.getLogger(__name__)

RETRY_WAITS_S = (2, 5, 10)
AGENT_BUDGET_S = 60
RETRIABLE = {500, 503}
MAX_CALL_S = 45                 # délai maximal d'un appel HTTP (toujours borné par le budget restant)
PROBE_TIMEOUT_S = 10
MODELS_FILE = CACHE_DIR / "gemini_models_ok.json"

CallFn = Callable[[str, str, str, dict | None, bool, float, float], GeminiReply]
#        (model, prompt, system, json_schema, use_search, temperature, timeout_s) -> GeminiReply


class ResilientGemini:
    """Même interface que GeminiClient pour agents.base.call_json : generate(), model, probe_log."""

    def __init__(self, call: CallFn, models: list[str], *, budget_s: float = AGENT_BUDGET_S,
                 waits=RETRY_WAITS_S, sleep=time.sleep, clock=time.monotonic):
        self.models = list(dict.fromkeys(m for m in models if m))
        if not self.models:
            raise GeminiConfigError("Aucun modèle Gemini à interroger.")
        self.call, self.budget_s, self.waits = call, budget_s, tuple(waits)
        self.sleep, self.clock = sleep, clock
        self.model = self.models[0]
        self.probe_log: list[str] = [f"modèles utilisables : {', '.join(self.models)}"]
        self.journal: list[str] = []      # tentatives, attentes, bascules (affiché et sauvegardé)
        self.deadline: float | None = None
        self.agent = ""

    def select_model(self) -> str:
        return self.model

    def start_agent(self, name: str):
        """Ouvre le budget de 60 s de l'agent suivant."""
        self.agent = name
        self.deadline = self.clock() + self.budget_s
        self.journal.append(f"— {name} : budget {self.budget_s:.0f} s")

    def _fail(self, why: str, errors: list[str]):
        self.journal.append(f"{self.agent} : échec ({why})")
        raise GeminiCallError(f"{why}.\n" + "\n".join(errors[-6:]))

    def generate(self, prompt: str, system: str, *, json_schema: dict | None = None,
                 use_search: bool = False, temperature: float = 0.4) -> GeminiReply:
        if self.deadline is None:
            self.start_agent("appel")
        errors: list[str] = []
        order = [self.model] + [m for m in self.models if m != self.model]
        for model in order:
            waits = list(self.waits)
            while True:
                remaining = self.deadline - self.clock()
                if remaining < 1:
                    self._fail(f"budget de {self.budget_s:.0f} s par agent épuisé", errors)
                try:
                    reply = self.call(model, prompt, system, json_schema, use_search, temperature,
                                      min(remaining, MAX_CALL_S))
                    if model != self.model:
                        self.journal.append(f"{self.agent} : bascule sur {model}")
                    self.model = model
                    self.journal.append(f"{self.agent} : réponse de {model}")
                    return reply
                except GeminiConfigError:
                    raise
                except Exception as e:  # noqa: BLE001
                    code = _code(e)
                    if code in (400, 401, 403) and "API key" in _short(e):
                        raise GeminiConfigError(f"Clé Gemini invalide ({code}) : {_short(e)}") from e
                    msg = _short(e)
                    prefix = f"{code} " if code and not msg.startswith(str(code)) else ("" if code else f"{type(e).__name__} ")
                    errors.append(f"{model}: {prefix}{msg}")
                    if code in RETRIABLE and waits:
                        w = waits.pop(0)
                        if self.clock() + w >= self.deadline:
                            self._fail(f"budget de {self.budget_s:.0f} s par agent épuisé", errors)
                        self.journal.append(f"{self.agent} : {model} {code}, nouvelle tentative dans {w} s")
                        log.warning("Gemini %s %s : nouvelle tentative dans %s s", model, code, w)
                        self.sleep(w)
                        continue
                    self.journal.append(f"{self.agent} : {model} abandonné ({code or type(e).__name__})")
                    break  # modèle suivant
        self._fail("aucun modèle Gemini n'a répondu", errors)


# ------------------------------------------------------------------ appel réel
def make_call(gc) -> CallFn:
    """Appel réel via le SDK google-genai d'un GeminiClient (un seul essai, délai borné)."""
    t = gc.types

    def call(model, prompt, system, json_schema, use_search, temperature, timeout_s):
        cfg = dict(system_instruction=system, temperature=temperature,
                   http_options=t.HttpOptions(timeout=max(1000, int(timeout_s * 1000))))
        if use_search:
            cfg["tools"] = [t.Tool(google_search=t.GoogleSearch())]
        elif json_schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = json_schema
        r = gc.client.models.generate_content(model=model, contents=prompt,
                                              config=t.GenerateContentConfig(**cfg))
        text = r.text or ""
        if not text.strip():
            raise GeminiCallError("réponse vide")
        return gc._reply(r, model, text)

    return call


def probe_models(gc, candidates: list[str] | None = None, timeout_s: float = PROBE_TIMEOUT_S) -> list[str]:
    """Teste les modèles EN PARALLÈLE (appel minimal, délai court) ; renvoie ceux qui répondent,
    dans l'ordre de préférence, et les enregistre dans data/cache/gemini_models_ok.json."""
    t = gc.types
    candidates = list(candidates or gc.candidates)
    cfg = t.GenerateContentConfig(http_options=t.HttpOptions(timeout=int(timeout_s * 1000)),
                                  max_output_tokens=5)

    def one(m):
        t0 = time.time()
        try:
            gc.client.models.generate_content(model=m, contents="Réponds OK.", config=cfg)
            return m, True, f"{m} : OK ({time.time() - t0:.1f} s)"
        except Exception as e:  # noqa: BLE001
            return m, False, f"{m} : indisponible ({_code(e) or type(e).__name__})"

    with ThreadPoolExecutor(max_workers=len(candidates) or 1) as ex:
        res = list(ex.map(one, candidates))
    ok = [m for m, good, _ in res if good]
    try:
        MODELS_FILE.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "ok": ok,
                                           "log": [x for _, _, x in res]}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return ok
