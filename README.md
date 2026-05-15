# Multi-Agent Hierarchical RL — Tesi v2

Implementazione del framework gerarchico di **Cipollone et al. (2025)** esteso al setting multi-agent.
Il framework usa **Algorithm 1** del paper: training MAPPO a tre livelli di astrazione, con il
V* del livello superiore che fornisce il segnale di reward shaping al livello sotto.

---

## Idea principale (per chi apre il progetto e vuole capire subito)

C'è una griglia 10×10 dove due agenti devono cooperare per salvare una vittima
(uno prende l'item, raggiungono insieme la vittima). Allenare MAPPO con sparse reward
direttamente su 10×10 **fallisce** — il segnale di reward è troppo debole.

La soluzione: costruiamo una gerarchia di astrazioni dello stesso ambiente fisico.

```
       LIVELLO 10 (concreto)             ← qui vogliamo che funzioni
            ▲
            │ shaping = V*_5 (proiezione 10→5)
            │
       LIVELLO 5 (medio)                  ← anche qui MAPPO impara
            ▲
            │ shaping = V*_3 (proiezione 5→3)
            │
       LIVELLO 3 (caso base)              ← questo è facile, MAPPO converge da solo
```

**Opzione A (confermata dal professore):** lo stesso episodio fisico viene visto a tre risoluzioni diverse.
L'episodio 100 di tutti gli esperimenti parte dalla stessa mappa concreta 10×10, proiettata al livello giusto.

---

## La narrativa della tesi: 5 esperimenti

```
Exp 1: Livello 10 SPARSE         →  fallisce          (sparse non basta al concreto)
Exp 2: Livello  5 SPARSE         →  fallisce          (sparse non basta neanche qui)
Exp 3: Livello  3 SPARSE         →  riesce            (caso base)
                                                       da qui otteniamo V*_3
Exp 4: Livello  5 + V*_3 shaping →  riesce            (con aiuto dal livello sotto)
                                                       da qui otteniamo V*_5
Exp 5: Livello 10 + V*_5 shaping →  riesce            (vittoria finale, è lo scopo della tesi)
```

I 5 esperimenti vengono fatti in sequenza dallo script `train_hierarchical.py`.
Alla fine viene generato un **grafico comparativo unico** con tutte e 5 le curve sovrapposte —
quella sarà la figura principale del capitolo della tesi.

---

## Struttura dei file

```
MultiAgent_Thesis/
├── README.md                              ← questo file
├── assets/                                ← le tue immagini PNG (sprites)
│   ├── robot1.png, robot2.png
│   ├── kit.png, victim.png, fire.png, wall.png (opzionale)
├── results/                               ← output del training (creato auto)
├── env/
│   ├── __init__.py
│   ├── map_generator.py                   ← genera mappa 10×10 + funzione di proiezione
│   ├── grid_world.py                      ← ambiente parametrizzato (size 3, 5, 10)
│   └── rescue_env_pz.py                   ← wrapper PettingZoo
└── training/
    ├── __init__.py
    ├── config.py                          ← TUTTI i parametri qui
    ├── networks.py                        ← Actor + Critic
    ├── buffer.py                          ← rollout buffer + GAE
    ├── mappo.py                           ← MAPPO con doppio learner (Algorithm 1)
    ├── abstract_layer.py                  ← VI analitica per V* + shaping
    ├── train_hierarchical.py              ← orchestratore: 5 esperimenti
    └── visualize_pygame.py                ← demo visiva di una policy
```

---

## Installazione

```bash
pip install torch numpy matplotlib pettingzoo gymnasium pygame
```

---

## Come lanciare il training

```bash
cd MultiAgent_Thesis
python -m training.train_hierarchical
```

Lo script lancia in sequenza i 5 esperimenti per ogni seed in `TRAIN["seeds"]`.
Per ogni esperimento stampa il progresso ogni 100 episodi e produce:
- un plot di convergenza
- file `.npy` con la storia di reward e success rate
- checkpoint `policy.pt` con i pesi della rete

Alla fine del seed produce il **plot comparativo** con tutte e 5 le curve.

### Tempi stimati (CPU singolo thread)

| Esperimento | Episodi | Max steps | Tempo |
|---|---|---|---|
| Exp 1 (L10 sparse) | 12000 | 200 | ~2 ore |
| Exp 2 (L5 sparse) | 5000 | 100 | ~30 min |
| Exp 3 (L3 sparse) | 2000 | 50 | ~10 min |
| Exp 4 (L5 hier) | 5000 | 100 | ~45 min |
| Exp 5 (L10 hier) | 12000 | 200 | ~2.5 ore |
| **Totale** | | | **~6 ore** |

Per testare velocemente che tutto funzioni, riduci il numero di episodi in `config.py`:
```python
TRAIN["episodes_per_level"] = {3: 200, 5: 300, 10: 500}
```

---

## Come modificare i parametri

Tutto in `training/config.py`. I principali:

```python
TRAIN["episodes_per_level"] = {
    3:  2000,    # episodi al livello 3
    5:  5000,    # al livello 5
    10: 12000,   # al livello 10
}

TRAIN["seeds"] = [42]   # aggiungi più seed per robustezza statistica

HIERARCHY["shaping_scale"] = 1.0           # scala del segnale di shaping
HIERARCHY["differential_shaping"] = True   # F = γV(s')-V(s) (potential-based)

CONVERGENCE["success_threshold"] = 0.80    # SR_last100 ≥ 80% = "risolto"
CONVERGENCE["failure_threshold"] = 0.30    # SR_last100 ≤ 30% = "fallito"
```

---

## Output del training

```
results/seed_42/
├── exp_1_level10_sparse/
│   ├── sr.npy                 ← success rate per episodio (sliding window 100)
│   ├── rewards.npy            ← reward per episodio
│   ├── policy.pt              ← checkpoint policy
│   └── training.png           ← plot di questo esperimento
├── exp_2_level5_sparse/       ← stessa struttura
├── exp_3_level3_sparse/
├── exp_4_level5_hier/
│   ├── ...
│   └── rewards_biased.npy     ← in più: reward con shaping (per il biased learner)
├── exp_5_level10_hier/        ← idem
└── comparison.png             ← LA figura principale: 5 curve sovrapposte
```

---

## Demo visiva

Dopo il training:

```bash
python -m training.visualize_pygame
```

Mostra l'agente che gira in tempo reale. Default: visualizza la policy del Esp 5 (livello 10 gerarchico).
Per visualizzare una policy diversa, modifica in fondo a `visualize_pygame.py`:

```python
EXPERIMENT = "exp_5_level10_hier"   # cambia in exp_3_level3_sparse, ecc.
```

Controlli da tastiera:
- **SPACE**: pausa / riprendi
- **N**: prossimo episodio (mappa nuova random)
- **R**: ricomincia episodio corrente
- **+ / -**: velocizza / rallenta
- **ESC** o **Q**: esci

---

## Decisioni di design (per la tesi)

Tutte le scelte sotto sono state confermate via mail dal professore.

1. **Algorithm 1 del paper**: training MAPPO a tutti e tre i livelli, non solo al concreto.
2. **V* single-agent ai livelli astratti**: la stessa V*(cella) è applicata indipendentemente
   ai due agenti come heuristica spaziale. Il multi-agent emerge dal training MAPPO,
   non dal V*.
3. **Reward strutturati**: manteniamo +5 pickup, +10 rescue, -0.3 muri, -5 fuochi (non goal MDP).
4. **Doppio learner di Algorithm 1**: il biased usa shaping, l'unbiased usa solo reward base.
   Le azioni vengono dal biased (esplora veloce); l'unbiased è la policy finale che converge
   all'ottimo (garanzia teorica del paper).
5. **Opzione A — stessa mappa a tre risoluzioni**: pre-generiamo un pool di seed di mappa, e tutti
   gli esperimenti li riusano nello stesso ordine. Episodio K di qualunque esperimento parte
   dalla stessa mappa fisica 10×10.
6. **V* tramite VI analitica**: ai livelli astratti V* è calcolato con Value Iteration esatta
   (le griglie sono piccole, ~9 e 25 stati). Più robusto e veloce di estrarlo dalla critic
   trainata; il critic alla convergenza darebbe lo stesso V*.

### Iterazione di tuning (dopo i primi risultati)

Dopo il primo training run abbiamo notato che Exp 5 (livello 10 gerarchico) non convergeva.
La diagnosi ha portato a 3 modifiche tecniche:

- **Proiezione ostacoli ibrida** (`map_generator.py`):
  - `10 → 5`: **any-is-blocked** (cella astratta bloccata se ALMENO UNA concreta lo è).
    Così V*_5 conosce muri e fuochi, e lo shaping al livello 10 guida AROUND gli ostacoli
    invece che attraverso.
  - `10 → 3` e `5 → 3`: **majority-is-blocked** (≥50%). "Any-is-blocked" sul 3×3 sarebbe
    troppo aggressivo (celle d'angolo coprono fino a 16 concrete) e bloccherebbe quasi tutto.
- **Fuochi nell'astrazione**: i fuochi del 10×10 contribuiscono al conteggio delle celle
  bloccate durante la proiezione. Prima venivano scartati al livello astratto, e V* astratto
  guidava gli agenti dritti sui fuochi.
- **Formula PBRS canonica**: `F = scale · (γ·V(s') - V(s))` invece di `scale · (V(s') - V(s))`.
  Aggiunge il fattore γ per coerenza teorica con Ng et al. (1999).
- **Scale aumentata**: `shaping_scale = 5.0` invece di `1.0`. Senza scaling, lo shaping
  era ~0.04 per step, invisibile rispetto a reward del task di +5/+10.

---

## Cosa aspettarsi nei risultati

Se tutto va bene, il plot comparativo dovrebbe mostrare questa storia:

- **Exp 1, 2** (sparse a livello 10 e 5): curve basse e piatte, sotto il 30%
- **Exp 3** (sparse a livello 3): sale rapidamente sopra l'80% (caso base risolto)
- **Exp 4** (hier a livello 5): sale a 70-80% grazie allo shaping da V*_3
- **Exp 5** (hier a livello 10): sale a 70-80% grazie allo shaping da V*_5

Il messaggio: **lo shaping derivato automaticamente dal livello superiore permette
a MAPPO di risolvere task che lo sparse reward non riesce a risolvere**.
Questo è l'automazione del reward shaping che il professore voleva vedere.

---

## Estensioni future

- Più seed per significatività statistica (cambia `TRAIN["seeds"]`)
- V* multi-agent (prodotto cartesiano delle posizioni) ai livelli astratti
- Conversione a goal MDP (Definizione 3 del paper) per coerenza teorica
- Estensione a un quarto livello più astratto (es. 2×2) per gerarchie più profonde
- Transizione all'ambiente continuo (Phase 2 della tesi)
