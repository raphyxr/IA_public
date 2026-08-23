import torch  # On charge PyTorch pour manipuler des tenseurs (tableaux de nombres). Exemple: torch.tensor([1, 2, 3])
import argparse  # Pour passer des options en ligne de commande. Exemple: --epochs 5
import os  # Outils systeme (ex: verifier si un fichier existe). Exemple: os.path.exists("modele_streaming.pt")
import json  # Pour lire vocab.json (mapping mot <-> id). Exemple: json.load(open("vocab.json","r",encoding="utf-8"))
import sys  # Pour mettre a jour la barre de chargement sur la meme ligne. Exemple: sys.stdout.write("\r...")
import torch.nn as nn  # On charge le module reseau de neurones. Exemple: nn.Linear(10, 5)
from torch.utils.data import TensorDataset, DataLoader  # Outils pour faire des mini-lots. Exemple: DataLoader(dataset, batch_size=32)

# Note: ne pas faire de print ici, sinon ca s'affiche a chaque "import modele_streaming" dans un notebook.

# Limites RAM (ajuste ces valeurs selon ta machine). Exemple: baisser VOCAB_LIMIT si ton PC manque de memoire.
VOCAB_LIMIT = 100000  # Taille max du vocabulaire apres compression. Exemple: ID 25000 devient 5000 si limite a 10000
EMB_DIM = 128  # Taille du vecteur embedding pour chaque mot/token. Exemple: un token devient un vecteur de 64 nombres
HIDDEN_DIM = 128  # Taille interne du Transformer feed-forward. Exemple: couche cache de taille 128
CONTEXT_SIZE = 128  # Nombre de tokens utilises pour predire le suivant. Exemple: [10, 11, 12] -> predire token suivant
BATCH_SIZE = 16  # Nombre d'exemples traites en meme temps. Exemple: 2 sequences par passage
CHUNK_TOKENS = 20000  # Nombre de tokens charges par morceau (chunk). Exemple: lire 5000 IDs puis entrainer
EPOCHS = 30  # Nombre de passages complets d'entrainement. Exemple: 1 = un seul tour sur les donnees lues
MAX_TOKENS_PER_EPOCH = None  # Limite de tokens lus par epoch. Exemple: arret apres 100000 tokens

# Reprise d'entrainement (resume). Exemple: si True et que le fichier existe, on recharge les poids avant de re-entrainer.
BASE_DIR = "/kaggle/input/datasets/bgbg1000/model-v-3/"  # Dossier Drive optionnel. Exemple: "/content/drive/MyDrive"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Dossier du script local. Exemple: "C:/.../IA"
DATA_DIR = BASE_DIR if BASE_DIR else SCRIPT_DIR  # Si BASE_DIR est vide, on lit les donnees dans le dossier du script. Exemple: ".../IA"
OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else SCRIPT_DIR  # Dossier ecriture. Kaggle: /input est lecture seule
DATASET_IDS_PATH = os.path.join(DATA_DIR, "dataset-ids.txt")  # Chemin du fichier d'IDs. Exemple: "/content/drive/MyDrive/dataset-i.txt"
DATASET_IDS_FALLBACK_PATH = os.path.join(DATA_DIR, "dataset_ids.txt")  # Autre nom possible du dataset. Exemple: ancienne version du fichier
CHECKPOINT_LOAD_PATH = os.path.join(SCRIPT_DIR, "modelV3.0.pt")  # Chemin du modele a CHARGER (lecture). Exemple: "C:/.../modele_streaming.pt"
CHECKPOINT_SAVE_PATH = os.path.join(OUTPUT_DIR, "modelV3.0.pt")  # Chemin du modele a SAUVEGARDER. Kaggle: "/kaggle/working/modelV3.0.pt"
RESUME_IF_CHECKPOINT_EXISTS = False  # True = reprend si possible, False = repart de zero. Exemple: False pour tout recommencer

# Affichage en mots (avec vocab.json). Exemple: afficher une "phrase" lisible au lieu des IDs.
VOCAB_JSON_PATH = os.path.join(DATA_DIR, "vocab.json")  # Chemin du vocab. Exemple: "/content/drive/MyDrive/vocab.json" (dict: mot -> id)
PRINT_PREDICTED_TEXT = True  # True = affiche la phrase + mots predits. Exemple: False = n'affiche que pred_id
PROMPT_TOKENS = 20  # Nombre de tokens de depart a afficher. Exemple: 20 mots au debut
GENERATE_TOKENS = 20  # Nombre de tokens a predire/apres. Exemple: ajoute 20 mots predits
PRINT_TRAINING_DECODE_EXAMPLE = False  # True = affiche un petit exemple decode pendant l'entrainement. Exemple: contexte -> mot predit / mot attendu
TRAINING_DECODE_EXAMPLES_PER_CHUNK = 10  # Nombre d'exemples affiches par chunk. Exemple: 1 = on montre seulement le premier exemple decode

def compress_id(token_id, vocab_limit=VOCAB_LIMIT):  # Fonction qui reduit un ID dans une plage fixe. Exemple: 12345 -> 2345 (si limite 10000)
    """Compresse un ID brut dans [0, vocab_limit-1] pour limiter la RAM."""  # Resume de la fonction. Exemple: evite un vocabulaire immense
    return token_id % vocab_limit  # Le modulo garde l'ID dans la plage voulue. Exemple: 10007 % 10000 = 7


def iter_ids(path=DATASET_IDS_PATH, chunk_chars=1_000_000):  # Generateur qui lit le fichier petit a petit. Exemple: ne lit pas tout en une fois
    """Lit les IDs par paquets de texte pour eviter de tout charger en RAM."""  # But: economiser la memoire. Exemple: lecture par blocs
    sd = 0
    with open(path, "r", encoding="utf-8") as f:  # Ouvre le fichier texte contenant les IDs. Exemple: "1 2 3 4 ..."
        reste = ""  # Stocke un bout de nombre coupe entre deux blocs. Exemple: "12" si le bloc finit au milieu
        while True:  # Boucle continue jusqu'a la fin du fichier. Exemple: lit bloc apres bloc
            bloc = f.read(chunk_chars)  # Lit un bloc de caracteres. Exemple: 1 000 000 caracteres
            if not bloc:  # Si rien n'est lu, on est a la fin. Exemple: chaine vide
                break  # On sort de la boucle. Exemple: fin de lecture

            texte = reste + bloc  # Recolle le bout precedent au bloc courant. Exemple: "12" + "34 56" -> "1234 56"
            parts = texte.split()  # Coupe le texte en elements separes par espace. Exemple: "10 20" -> ["10", "20"]

            if texte and not texte[-1].isspace():  # Si le bloc se termine au milieu d'un nombre. Exemple: finit par "123"
                reste = parts.pop() if parts else texte  # On garde le dernier bout incomplet pour le prochain bloc. Exemple: garde "123"
            else:  # Sinon, le bloc finit proprement sur un espace. Exemple: "10 20 "
                reste = ""  # Rien a garder. Exemple: vide
            
            for p in parts:  # Parcourt chaque ID texte complet du bloc. Exemple: p="10", puis p="20"
                yield compress_id(int(p))
                sd = sd + 1
                if sd == 200:
                    sd = 0
                # Convertit en int, compresse, puis renvoie un ID. Exemple: "20" -> 20 -> 20
            # debug print removed
        if reste:  # A la fin, s'il reste un dernier nombre incomplet devenu complet. Exemple: reste="789"
            yield compress_id(int(reste))  # On le convertit et on le renvoie aussi. Exemple: 789 -> 789


def load_ids(path=DATASET_IDS_PATH, max_ids=None):  # Charge des IDs en liste Python (utile pour test rapide). Exemple: max_ids=4 -> [10, 11, 12, 13]
    """Version compatible: charge un nombre limite d'IDs (ou tous si max_ids=None)."""  # Resume du comportement. Exemple: sans limite = tous
    ids = []  # Liste qui va contenir les IDs. Exemple: [] puis [10] puis [10, 11]
    for i, tok in enumerate(iter_ids(path)):  # Lit les IDs un par un avec index. Exemple: i=0 tok=10
        ids.append(tok)  # Ajoute l'ID lu dans la liste. Exemple: ajoute 10
        if max_ids is not None and i + 1 >= max_ids:  # Stop si on a atteint la limite demandee. Exemple: max_ids=100
            break  # Arret de la lecture. Exemple: sort de la boucle
    return ids  # Renvoie la liste finale. Exemple: [10, 11, 12]


def choisir_dataset_ids_path(primary_path, fallback_path=None, min_ids=1):  # Trouve un fichier dataset qui contient assez d'IDs. Exemple: essaie dataset-i puis dataset_ids
    """Choisit le premier fichier dataset lisible contenant au moins min_ids IDs."""
    candidats = [primary_path]  # Liste des chemins a tester. Exemple: ["dataset-i.txt"]
    if fallback_path and fallback_path != primary_path:  # Ajoute le fallback seulement s'il est different. Exemple: dataset_ids.txt
        candidats.append(fallback_path)  # Ajoute l'ancien nom possible. Exemple: ["dataset-i.txt", "dataset_ids.txt"]

    diagnostics = []  # Messages utiles si aucun fichier ne convient. Exemple: fichier vide, absent, etc.
    for path in candidats:  # Teste chaque chemin dans l'ordre. Exemple: d'abord dataset-i.txt
        if not os.path.exists(path):  # Si le fichier n'est pas present. Exemple: mauvais Drive
            diagnostics.append(f"{path}: absent")
            continue

        size = os.path.getsize(path)  # Taille en octets. Exemple: 1049176251
        print(f"dataset trouve: {path} ({size} octets)")  # Affiche ce que Colab voit vraiment. Exemple: utile si fichier vide
        try:
            ids_sample = load_ids(path, max_ids=min_ids)  # Essaie de lire juste quelques IDs. Exemple: 9 IDs
        except Exception as exc:
            diagnostics.append(f"{path}: lecture impossible ({type(exc).__name__}: {exc})")
            continue

        if len(ids_sample) >= min_ids:  # Si le fichier contient assez d'IDs pour demarrer. Exemple: 9 IDs lus
            return path, ids_sample

        diagnostics.append(f"{path}: {len(ids_sample)} ID(s) lus, {min_ids} requis, taille {size} octets")

    raise ValueError(
        "Aucun dataset utilisable trouve. "
        + " | ".join(diagnostics)
        + ". Dans Colab, verifie que le fichier n'est pas vide dans Drive ou utilise le bon chemin BASE_DIR."
    )


def embedding(ids, emb_layer):  # Transforme des IDs en vecteurs denses. Exemple: [10, 11, 12] -> 3 vecteurs
    """Transforme des IDs (torch.long) en vecteurs d'embedding."""  # Resume de la fonction. Exemple: chaque ID devient un vecteur
    return emb_layer(ids)  # Appelle la couche embedding. Exemple: shape [batch, temps, EMB_DIM]


def afficher_infos_device(device):  # Affiche clairement si le GPU est utilise. Exemple: cuda:0 NVIDIA...
    """Affiche le device PyTorch utilise et quelques infos CUDA si disponibles."""
    print(f"device utilise: {device}")
    print(f"cuda disponible: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
        print(f"cuda version pytorch: {torch.version.cuda}")
        print(f"nombre de gpu detectes: {torch.cuda.device_count()}")
    else:
        print("attention: entrainement sur CPU (GPU absent ou CUDA inutilisable avec cette installation PyTorch)")


def choisir_device_compatible():  # Choisit CUDA seulement si PyTorch peut vraiment executer un kernel dessus.
    """Retourne cuda si utilisable, sinon cpu avec un diagnostic clair."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    cuda_device = torch.device("cuda")
    try:
        gpu_name = torch.cuda.get_device_name(cuda_device)
        capability = torch.cuda.get_device_capability(cuda_device)
        print(f"gpu detecte: {gpu_name} (compute capability sm_{capability[0]}{capability[1]})")
        print(f"cuda version pytorch: {torch.version.cuda}")
        if capability[0] < 7:
            print("GPU non utilise: cette installation PyTorch ne supporte pas les GPU sm_60 comme la Tesla P100.")
            print("bascule sur CPU. Sur Kaggle, choisis un GPU plus recent comme T4, ou une image PyTorch compatible P100.")
            return torch.device("cpu")

        # Teste un vrai kernel CUDA. Sur Kaggle/P100, certaines versions recentes de PyTorch
        # voient CUDA mais ne contiennent pas de kernel pour sm_60.
        test_tensor = torch.ones(1, device=cuda_device)
        _ = test_tensor + 1
        torch.cuda.synchronize(cuda_device)
        return cuda_device
    except Exception as exc:
        print("CUDA detecte mais inutilisable avec cette installation PyTorch.")
        print(f"raison: {type(exc).__name__}: {exc}")
        print("bascule sur CPU. Sur Kaggle, utilise une image PyTorch compatible P100/sm_60 ou un GPU plus recent.")
        return torch.device("cpu")


def verifier_modules_sur_device(device, *modules):  # Verifie que tous les modules sont sur le bon device. Exemple: emb/fc sur cuda
    """Leve une erreur claire si un module est reste sur CPU alors qu'on attend CUDA."""
    expected_device = torch.device(device)
    for module in modules:
        for param in module.parameters():
            param_device = param.device
            same_type = param_device.type == expected_device.type
            same_index = expected_device.index is None or param_device.index == expected_device.index
            if not same_type or not same_index:
                raise RuntimeError(f"module sur {param_device}, attendu {expected_device}")
            break


def charger_vocab_pour_ids_compresses(vocab_json_path, vocab_limit=VOCAB_LIMIT):  # Charge vocab.json et prepare un tableau id->mot pour les IDs compresses
    """Charge vocab.json (mot->id) et fabrique une table id_compresse -> mot (approximatif si modulo)."""  # Important: avec modulo, ce n'est pas parfait
    if not os.path.exists(vocab_json_path):  # Si le fichier n'existe pas. Exemple: vocab.json absent
        return None, 0, 0  # Rien a charger. Exemple: (None, 0, 0)

    try:
        with open(vocab_json_path, "r", encoding="utf-8") as f:  # Ouvre le vocab en UTF-8. Exemple: gros JSON
            token_to_id = json.load(f)  # Lit tout le JSON. Exemple: {"les":1, "chat":2, ...}
    except json.JSONDecodeError as exc:
        print(
            "vocab.json invalide: "
            f"{vocab_json_path} ligne {exc.lineno}, colonne {exc.colno} "
            f"(position {exc.pos}). Decodage texte desactive."
        )
        print("Regenerer ou recopier vocab.json, puis verifier le fichier avec: python -m json.tool vocab.json")
        return None, 0, 0

    id_to_token = ["<unk>"] * int(vocab_limit)  # Tableau taille vocab_limit. Exemple: 10000 cases
    collisions = 0  # Compte quand 2 mots tombent sur le meme id compresse. Exemple: collision si vocab > vocab_limit
    max_raw_id = -1  # Plus grand ID vu dans le vocab. Exemple: 500000

    if isinstance(token_to_id, dict):  # Le format attendu: dict mot -> id. Exemple: {"les": 1}
        for token, raw_id in token_to_id.items():  # Parcourt chaque mot et son ID. Exemple: token="les", raw_id=1
            try:  # Au cas ou raw_id n'est pas un int. Exemple: "123" en string
                raw_id_int = int(raw_id)  # Convertit en int. Exemple: "123" -> 123
            except Exception:  # Si conversion impossible. Exemple: raw_id="abc"
                continue  # On ignore cette entree. Exemple: skip

            if raw_id_int > max_raw_id:  # Met a jour le max. Exemple: 10 puis 200 puis 500000
                max_raw_id = raw_id_int  # Sauve le nouveau max. Exemple: 500000

            compressed = raw_id_int % int(vocab_limit)  # Meme compression que le dataset. Exemple: 12345 % 10000 = 2345
            if id_to_token[compressed] == "<unk>":  # Premiere fois qu'on voit cet ID compresse. Exemple: vide avant
                id_to_token[compressed] = str(token)  # On garde ce mot. Exemple: id_to_token[1]="les"
            else:  # Sinon, on a deja un mot pour cet ID. Exemple: collision
                collisions += 1  # Compte la collision. Exemple: +1

    return id_to_token, max_raw_id, collisions  # Renvoie la table + infos. Exemple: (liste, 500000, 120000)


def ids_vers_phrase(ids, id_to_token):  # Convertit une liste d'IDs en phrase (texte). Exemple: [1,2,3] -> "les chats mangent"
    if not id_to_token:  # Si pas de vocab charge. Exemple: None
        return " ".join(str(int(i)) for i in ids)  # Fallback: affiche les IDs. Exemple: "1 2 3"

    mots = []  # Liste de mots decodes. Exemple: ["les","bains"]
    for i in ids:  # Pour chaque ID. Exemple: i=1 puis i=2
        idx = int(i)  # Assure un int. Exemple: tensor(1) -> 1
        if 0 <= idx < len(id_to_token):  # Si dans la table. Exemple: idx=500 < 10000
            mots.append(id_to_token[idx])  # Ajoute le mot. Exemple: "les"
        else:  # Si hors limites. Exemple: idx negatif ou trop grand
            mots.append(f"<id:{idx}>")  # Marqueur simple. Exemple: "<id:123456>"
    return " ".join(mots)  # Retourne la phrase. Exemple: "les bains formatnum"


def afficher_exemple_decode(X_batch, y_batch, pred_ids, id_to_token, max_examples=1):  # Affiche quelques exemples lisibles a partir d'un batch. Exemple: contexte="je suis" -> predit="ici" / attendu="la"
    if not id_to_token:  # Si on n'a pas de vocabulaire, on ne peut pas convertir proprement les IDs en mots. Exemple: vocab.json absent
        return  # On sort sans rien afficher. Exemple: evite une erreur ou un affichage confus

    nb_exemples = min(int(max_examples), len(pred_ids))  # Limite le nombre d'exemples affiches a ce qui existe vraiment dans le batch. Exemple: min(1, 2) -> 1
    for i in range(nb_exemples):  # Parcourt seulement les premiers exemples choisis. Exemple: i=0 puis stop si max_examples=1
        contexte_ids = X_batch[i].tolist()  # Recupere la sequence de contexte sous forme de liste Python. Exemple: [10, 11, 12, 13]
        attendu_id = int(y_batch[i].item())  # Recupere l'ID reel attendu pour cet exemple. Exemple: 14
        predit_id = int(pred_ids[i].item())  # Recupere l'ID predit par le modele. Exemple: 22

        contexte_texte = ids_vers_phrase(contexte_ids, id_to_token)  # Convertit les IDs du contexte en mots. Exemple: "je suis a"
        attendu_texte = ids_vers_phrase([attendu_id], id_to_token)  # Convertit le vrai prochain ID en mot lisible. Exemple: "paris"
        predit_texte = ids_vers_phrase([predit_id], id_to_token)  # Convertit l'ID predit en mot lisible. Exemple: "lyon"

        print(f"exemple decode {i + 1}:")  # Affiche un petit titre pour mieux separer les exemples. Exemple: exemple decode 1:
        print(f"  contexte : {contexte_texte}")  # Montre les mots donnes au modele. Exemple: contexte : je suis a
        print(f"  attendu  : {attendu_texte} (id={attendu_id})")  # Montre la bonne reponse. Exemple: attendu : paris (id=14)
        print(f"  predit   : {predit_texte} (id={predit_id})")  # Montre ce que le modele a choisi. Exemple: predit : lyon (id=22)
        # Donne une mini explication pedagogique. Exemple: phrase simple


def creer_exemples(ids_liste, context_size=3):  # Cree les entrees X et cibles y pour l'entrainement. Exemple: [1,2,3,4] -> X=[[1,2,3]], y=[4]
    X = []  # Liste des contextes (entrees). Exemple: [[10,11,12], [11,12,13]]
    y = []  # Liste des cibles (sorties attendues). Exemple: [13, 14]

    for i in range(len(ids_liste) - context_size):  # Glisse une fenetre sur la liste. Exemple: i=0 puis i=1...
        contexte = ids_liste[i:i + context_size]  # Prend context_size tokens pour l'entree. Exemple: [10, 11, 12]
        cible = ids_liste[i + context_size]  # Prend le token suivant comme cible. Exemple: 13
        X.append(contexte)  # Ajoute le contexte dans X. Exemple: X += [[10,11,12]]
        y.append(cible)  # Ajoute la cible dans y. Exemple: y += [13]

    X = torch.tensor(X, dtype=torch.long)  # Convertit X en tenseur PyTorch. Exemple: shape [nb_exemples, context_size]
    y = torch.tensor(y, dtype=torch.long)  # Convertit y en tenseur PyTorch. Exemple: shape [nb_exemples]
    return X, y  # Renvoie entrees et cibles. Exemple: (X, y)


def batches(X, y, batch_size=32, shuffle=True):  # Prepare des mini-lots pour l'entrainement. Exemple: batch_size=32
    dataset = TensorDataset(X, y)  # Colle X et y dans un dataset. Exemple: element i -> (X[i], y[i])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # Cree le chargeur par lot. Exemple: renvoie 32 exemples par iteration
    return loader  # Renvoie l'objet iterable DataLoader. Exemple: for X_batch, y_batch in loader


def afficher_barre_progression(epoch, chunk_index, batch_index, total_batches, loss=None, largeur=30):  # Affiche l'avancement d'un chunk. Exemple: [####------] 4/20
    total_batches = max(int(total_batches), 1)  # Evite une division par zero. Exemple: total 0 -> 1
    ratio = min(max(batch_index / total_batches, 0.0), 1.0)  # Pourcentage entre 0 et 1. Exemple: 0.5
    remplis = int(ratio * largeur)  # Nombre de caracteres remplis dans la barre. Exemple: 15 sur 30
    barre = "#" * remplis + "-" * (largeur - remplis)  # Construit la barre texte. Exemple: ####------
    loss_txt = f" loss={loss:.4f}" if loss is not None else ""  # Ajoute la loss si disponible. Exemple: loss=2.3456
    sys.stdout.write(  # \r revient au debut de la ligne pour rafraichir la meme barre. Exemple: pas une ligne par batch
        f"\repoch {epoch} | chunk {chunk_index} [{barre}] "
        f"{batch_index}/{total_batches} ({ratio * 100:5.1f}%){loss_txt}"
    )
    sys.stdout.flush()  # Force l'affichage immediat. Exemple: utile dans certains terminaux
    if batch_index >= total_batches:  # A la fin du chunk, on passe a la ligne suivante. Exemple: barre terminee
        print()


# def prediction(X_batch, y_batch, emb_layer, transformer, fc, optimizer, criterion):  # Fait un passage avant + arriere pour un batch. Exemple: une etape d'apprentissage
#     # deplacer les batches sur le meme device que le modele (GPU si present)
#     device = next(emb_layer.parameters()).device# Convertit IDs en vecteurs. Exemple: [B, T] -> [B, T, EMB_DIM]
   
#     X_batch = X_batch.to(device)
#     y_batch = y_batch.to(device)

#     vecs = embedding(X_batch, emb_layer)
#     out = transformer(vecs)
#     last = out[:, -1, :]  
#     logits = fc(last)  # Transforme en scores sur tout le vocabulaire. Exemple: [B, VOCAB_LIMIT]
#     loss = criterion(logits, y_batch)  # Calcule l'erreur entre prediction et vraie cible. Exemple: CrossEntropy

#     optimizer.zero_grad()  # Remet les gradients a zero avant backprop. Exemple: evite accumulation des gradients
#     loss.backward()  # Calcule les gradients par retropropagation. Exemple: derivees de chaque poids
#     optimizer.step()  # Met a jour les poids du modele. Exemple: descente de gradient SGD

#     pred_ids = torch.argmax(logits, dim=-1)  # Prend l'ID avec le score le plus haut. Exemple: [52, 17]
#     return loss.item(), pred_ids  # Renvoie la perte (nombre) et les IDs predits. Exemple: (3.12, tensor([52,17]))

def prediction(X_batch, y_batch, emb_layer, pos_embedding, transformer, fc, optimizer, criterion):
    device = next(emb_layer.parameters()).device
    X_batch = X_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)

    positions = torch.arange(X_batch.size(1), device=device).unsqueeze(0)
    vecs = embedding(X_batch, emb_layer) + pos_embedding(positions)

    out = transformer(vecs)

    # logits for each position
    logits = fc(out)
    # predictions for the last position
    logits_last = logits[:, -1, :]
    pred_ids = torch.argmax(logits_last, dim=-1)

    # compute loss only on the last token (standard next-token training)
    loss = criterion(logits_last, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), pred_ids

def init_training_objects(vocab_size=VOCAB_LIMIT, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, lr=0.001, optimizer_name="sgd"):
    """Cree le modele + loss + optimizer (utile pour entrainer plusieurs fois dans le meme notebook)."""

    emb_layer = nn.Embedding(vocab_size, emb_dim)
    pos_embedding = nn.Embedding(CONTEXT_SIZE, emb_dim)
    transformer_layer = nn.TransformerEncoderLayer(
        d_model=emb_dim,
        nhead=4,
        dim_feedforward=hidden_dim,
        batch_first=True,
    )
    transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
    fc = nn.Linear(emb_dim, vocab_size)
    criterion = nn.CrossEntropyLoss()

    params = (
        list(emb_layer.parameters())
        + list(pos_embedding.parameters())
        + list(transformer.parameters())
        + list(fc.parameters())
    )

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        optimizer = torch.optim.SGD(params, lr=lr)
    return emb_layer, pos_embedding, transformer, fc, optimizer, criterion


def load_checkpoint_if_compatible(path, emb_layer, pos_embedding, transformer, fc, optimizer=None, vocab_size=None, emb_dim=None, hidden_dim=None):
    """Charge un checkpoint seulement si les tailles correspondent, sinon leve une erreur claire."""
    ckpt = torch.load(path, map_location="cpu")

    meta = ckpt.get("meta")
    if isinstance(meta, dict):
        expected = {}
        if vocab_size is not None:
            expected["vocab_size"] = vocab_size
        if emb_dim is not None:
            expected["emb_dim"] = emb_dim
        if hidden_dim is not None:
            expected["hidden_dim"] = hidden_dim

        mismatches = []
        for k, v in expected.items():
            if meta.get(k) != v:
                mismatches.append(f"{k}: ckpt={meta.get(k)} vs actuel={v}")
        if mismatches:
            raise ValueError("checkpoint incompatible (tailles differentes). " + " | ".join(mismatches))
    else:
        # Ancien checkpoint: pas de meta -> on compare au moins Embedding et Linear.
        emb_w = ckpt.get("emb", {}).get("weight")
        fc_w = ckpt.get("fc", {}).get("weight")
        if emb_w is not None and tuple(emb_w.shape) != tuple(emb_layer.weight.shape):
            raise ValueError(f"checkpoint incompatible (embedding): ckpt={tuple(emb_w.shape)} vs actuel={tuple(emb_layer.weight.shape)}")
        if fc_w is not None and tuple(fc_w.shape) != tuple(fc.weight.shape):
            raise ValueError(f"checkpoint incompatible (fc): ckpt={tuple(fc_w.shape)} vs actuel={tuple(fc.weight.shape)}")

    emb_layer.load_state_dict(ckpt["emb"])
    pos_embedding.load_state_dict(ckpt["pos_embedding"])
    transformer.load_state_dict(ckpt["transformer"])
    fc.load_state_dict(ckpt["fc"])
    if optimizer is not None and "optimizer" in ckpt:
        opt_state = ckpt["optimizer"]
        opt_groups = opt_state.get("param_groups", []) if isinstance(opt_state, dict) else []
        if isinstance(optimizer, torch.optim.Adam) and any("betas" not in group for group in opt_groups):
            print("optimizer checkpoint incompatible avec Adam: etat optimiseur ignore")
        else:
            optimizer.load_state_dict(opt_state)

def save_checkpoint(path, emb_layer, pos_embedding, transformer, fc, optimizer=None, meta=None):
    """Sauvegarde un checkpoint (utile pour reprendre plus tard).

    IMPORTANT: torch.save() n'est PAS atomique. Si on lui donne directement
    le chemin final, il tronque le fichier existant a 0 octet AVANT d'ecrire
    les nouvelles donnees. Si le processus est coupe pendant cette ecriture
    (crash, coupure GPU, limite de session, OOM...), le fichier .pt reste
    vide ou corrompu, meme si un ancien checkpoint valide existait avant.

    Solution: on ecrit dans un fichier temporaire a cote, on force l'ecriture
    sur le disque (flush + fsync), puis on renomme (os.replace) vers le
    chemin final. Un renommage est atomique au niveau du systeme de fichiers:
    soit l'ancien fichier reste intact, soit le nouveau est complet. Jamais
    d'etat "a moitie ecrit".
    """
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "emb": emb_layer.state_dict(),
        "pos_embedding": pos_embedding.state_dict(),
        "transformer": transformer.state_dict(),
        "fc": fc.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if meta is not None:
        payload["meta"] = meta

    tmp_path = path + ".tmp"  # Fichier temporaire ecrit a cote. Exemple: "modelV3.0.pt.tmp"
    try:
        with open(tmp_path, "wb") as f:  # Ouvre le fichier temporaire en ecriture binaire. Exemple: cree modelV3.0.pt.tmp
            torch.save(payload, f)  # Ecrit le checkpoint dans le temporaire (pas encore le fichier final)
            f.flush()  # Force Python/le buffer a transmettre les octets au systeme d'exploitation
            os.fsync(f.fileno())  # Force l'OS a ecrire physiquement sur le disque (pas juste en cache memoire)
    except Exception:
        if os.path.exists(tmp_path):  # Si l'ecriture temporaire a echoue en cours de route. Exemple: coupure pendant l'ecriture
            os.remove(tmp_path)  # On supprime le temporaire incomplet, le fichier final (ancien) reste intact
        raise  # On relance l'erreur pour ne pas cacher le probleme

    if os.path.exists(path):  # Garde une copie de secours de l'ancien checkpoint valide. Exemple: modelV3.0.pt -> modelV3.0.pt.bak
        backup_path = path + ".bak"
        try:
            os.replace(path, backup_path)  # Renomme l'ancien fichier en .bak (ecrase l'ancien .bak s'il existe)
        except OSError:
            pass  # Si ca echoue (rare), on continue quand meme: le principal est de ne pas perdre le nouveau checkpoint

    os.replace(tmp_path, path)  # Renommage atomique: le fichier final apparait complet ou n'apparait pas du tout

    saved_size = os.path.getsize(path)  # Verifie la taille reelle du fichier ecrit sur disque. Exemple: 5243180 octets
    if saved_size == 0:  # Securite supplementaire: si jamais le fichier final est vide, on previent clairement
        raise IOError(f"echec de sauvegarde: {path} fait 0 octet apres ecriture")
    return saved_size  # Renvoie la taille pour que l'appelant puisse l'afficher/logger


def build_checkpoint_meta(vocab_size, emb_dim, hidden_dim, context_size, epoch=None, chunk_index=None, seen_tokens=None):
    """Construit les metadonnees du checkpoint pour faciliter la reprise et le suivi."""
    meta = {
        "vocab_size": vocab_size,
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "context_size": context_size,
    }
    if epoch is not None:
        meta["epoch"] = epoch
    if chunk_index is not None:
        meta["chunk_index"] = chunk_index
    if seen_tokens is not None:
        meta["seen_tokens"] = seen_tokens
    return meta


def train_streaming(  # Fonction principale d'entrainement en lecture progressive (streaming). Exemple: train sans charger tout le dataset
    path,  # Chemin du fichier d'IDs. Exemple: "/content/drive/MyDrive/dataset_ids.txt"
    emb_layer,  # Couche embedding du modele. Exemple: nn.Embedding(...)
    pos_embedding,  # Embeddings qui indiquent la position des tokens
    transformer,  # Encodeur Transformer. Exemple: nn.TransformerEncoder(...)
    fc,  # Couche lineaire de sortie. Exemple: nn.Linear(...)
    optimizer,  # Optimiseur qui met a jour les poids. Exemple: torch.optim.SGD(...)
    criterion,  # Fonction de perte. Exemple: nn.CrossEntropyLoss()
    id_to_token=None,  # Table optionnelle pour decoder les IDs en mots. Exemple: ["<unk>", "bonjour", "chat", ...]
    context_size=3,  # Taille du contexte utilise pour predire. Exemple: 3
    batch_size=32,  # Taille des mini-lots. Exemple: 32
    chunk_tokens=200_000,  # Taille d'un morceau de donnees lu avant entrainement. Exemple: 200000 tokens
    max_tokens_per_epoch=None,  # Limite de tokens a lire pendant cette epoch. Exemple: 100000
    print_decoded_example=False,  # True = affiche un exemple decode pendant le train. Exemple: contexte -> predit / attendu
    decoded_examples_per_chunk=100,  # Nombre max d'exemples a afficher par chunk. Exemple: 1
    checkpoint_path=None,  # Chemin du fichier .pt a sauvegarder apres chaque chunk. Exemple: "modele_streaming.pt"
    checkpoint_meta=None,  # Metadonnees communes du checkpoint. Exemple: tailles du modele
):  # Fin des parametres de la fonction. Exemple: appel train_streaming(...)
    """Entraine en streaming: lit le fichier par paquets d'IDs, puis entraine chunk par chunk."""  # Resume du flux. Exemple: lecture->train->lecture->train
    emb_layer.train()
    pos_embedding.train()
    transformer.train()
    fc.train()
    device = next(emb_layer.parameters()).device
    verifier_modules_sur_device(device, emb_layer, pos_embedding, transformer, fc)


    total_loss = 0.0  # Somme des pertes de tous les batches. Exemple: 0.0 puis 2.1 puis 4.8
    total_batches = 0  # Compteur de mini-lots traites. Exemple: 0 puis 1 puis 2

    buffer_ids = []  # Tampon temporaire d'IDs du chunk courant. Exemple: [10,11,12,...]
    seen_tokens = 0  # Nombre total de tokens lus jusque la. Exemple: 5000
    chunk_index = 0  # Numero du chunk traite. Exemple: 1er chunk, 2e chunk
    epoch_num = checkpoint_meta.get("epoch", "?") if isinstance(checkpoint_meta, dict) else "?"  # Numero d'epoch pour l'affichage. Exemple: 1
    for tok in iter_ids(path):  # Lit chaque token compresse depuis le fichier. Exemple: tok=10, puis 11, etc.
        buffer_ids.append(tok)  # Ajoute le token au tampon. Exemple: buffer grandit
        seen_tokens += 1  # Incremente le compteur global. Exemple: 42 -> 43

        if max_tokens_per_epoch is not None and seen_tokens >= max_tokens_per_epoch:  # Stop si limite atteinte. Exemple: a 100000 tokens
            break  # Sort de la lecture pour finir l'epoch. Exemple: fin anticipee

        if len(buffer_ids) >= chunk_tokens:  # Si le tampon a atteint la taille chunk, on entraine dessus. Exemple: 5000 tokens
            chunk_index += 1  # Passe au numero de chunk suivant. Exemple: 0 -> 1
            if len(buffer_ids) > context_size:  # Verifie qu'on a assez de tokens pour creer au moins un exemple. Exemple: >3
                X, y = creer_exemples(buffer_ids, context_size=context_size)  # Construit les paires (contexte, cible). Exemple: X=[[...]], y=[...]
                loader = batches(X, y, batch_size=batch_size, shuffle=True)  # Cree les mini-lots du chunk. Exemple: lots de 2 si batch_size=2
                total_batches_chunk = len(loader)  # Nombre de mini-lots dans ce chunk. Exemple: 2500
                exemple_decode_affiche = False  # Sert a n'afficher qu'un petit nombre d'exemples par chunk. Exemple: False puis True apres le premier affichage
                for batch_index, (X_batch, y_batch) in enumerate(loader, start=1):  # Parcourt chaque mini-lot du chunk. Exemple: batch 1, batch 2, ...
                    loss, pred_ids = prediction(X_batch, y_batch, emb_layer, pos_embedding, transformer, fc, optimizer, criterion)  # Fait l'apprentissage du mini-lot et recupere les IDs predits
                    total_loss += loss  # Ajoute la perte du batch au total. Exemple: 4.8 + 1.2
                    total_batches += 1  # Compte un batch de plus. Exemple: 10 -> 11
                    afficher_barre_progression(epoch_num, chunk_index, batch_index, total_batches_chunk, loss=loss)  # Met a jour la barre du chunk courant. Exemple: 42%
                    if print_decoded_example and not exemple_decode_affiche:  # Affiche seulement un petit exemple decode pour ce chunk. Exemple: sur le premier batch uniquement
                        print()  # Separe l'exemple decode de la barre de progression. Exemple: evite de melanger les lignes
                        afficher_exemple_decode(X_batch, y_batch, pred_ids, id_to_token, max_examples=decoded_examples_per_chunk)  # Convertit les IDs du batch en mots lisibles
                        exemple_decode_affiche = True  # Evite d'afficher le meme type d'exemple a chaque batch. Exemple: passe a True
            if checkpoint_path:  # Sauvegarde un checkpoint apres chaque chunk pour ne pas perdre la progression. Exemple: modele_streaming.pt
                meta = dict(checkpoint_meta or {})  # Copie les infos communes pour y ajouter l'avancement courant. Exemple: tailles + epoch
                meta["chunk_index"] = chunk_index  # Note le chunk qui vient d'etre termine. Exemple: 7
                meta["seen_tokens"] = seen_tokens  # Note combien de tokens ont ete lus. Exemple: 35000
                taille = save_checkpoint(checkpoint_path, emb_layer, pos_embedding, transformer, fc, optimizer=optimizer, meta=meta)  # Ecrit le fichier .pt sur disque (ecriture atomique)
                print(f"checkpoint sauvegarde: {checkpoint_path} ({taille} octets)")  # Confirme la sauvegarde ET la taille reelle sur disque. Exemple: checkpoint sauvegarde: modelV3.0.pt (5243180 octets)
            print(f"chunk {chunk_index} traite, tokens vus: {seen_tokens}")  # Affiche la progression. Exemple: chunk 1 traite, tokens vus: 5000
            buffer_ids = buffer_ids[-context_size:]  # Garde seulement les derniers tokens pour continuer proprement. Exemple: garde 3 derniers

    if len(buffer_ids) > context_size:  # Apres la boucle, traite le dernier morceau restant. Exemple: fin de fichier
        chunk_index += 1  # Compte aussi le dernier petit morceau comme un chunk complet pour le suivi. Exemple: 20 -> 21
        X, y = creer_exemples(buffer_ids, context_size=context_size)  # Cree les exemples du reste. Exemple: dernier petit lot
        loader = batches(X, y, batch_size=batch_size, shuffle=True)  # Prepare les mini-lots du reste. Exemple: 1 ou plusieurs batches
        total_batches_chunk = len(loader)  # Nombre de mini-lots dans le dernier chunk. Exemple: 17
        exemple_decode_affiche = False  # Repart a zero pour le petit morceau final. Exemple: False au debut du reste
        for batch_index, (X_batch, y_batch) in enumerate(loader, start=1):  # Entraine sur ce dernier morceau. Exemple: boucle finale
            loss, pred_ids = prediction(X_batch, y_batch, emb_layer, pos_embedding, transformer, fc, optimizer, criterion)
  # Mise a jour du modele avec recuperation des IDs predits
            total_loss += loss  # Ajoute la perte. Exemple: +0.9
            total_batches += 1  # Compte le batch. Exemple: +1
            afficher_barre_progression(epoch_num, chunk_index, batch_index, total_batches_chunk, loss=loss)  # Met a jour la barre du dernier chunk. Exemple: 100%
            if print_decoded_example and not exemple_decode_affiche:  # Affiche un exemple decode aussi pour la fin si besoin. Exemple: premier batch du reste
                print()  # Separe l'exemple decode de la barre de progression. Exemple: sortie plus lisible
                afficher_exemple_decode(X_batch, y_batch, pred_ids, id_to_token, max_examples=decoded_examples_per_chunk)  # Montre le contexte, le mot attendu et le mot predit
                exemple_decode_affiche = True  # Evite d'en afficher trop. Exemple: stop apres le premier
        if checkpoint_path:  # Sauvegarde aussi le dernier morceau final si present. Exemple: reste de fin de fichier
            meta = dict(checkpoint_meta or {})  # Copie les infos communes. Exemple: tailles du modele
            meta["chunk_index"] = chunk_index  # Note ce dernier chunk de fin. Exemple: 21
            meta["seen_tokens"] = seen_tokens  # Conserve le nombre total de tokens lus dans l'epoch. Exemple: 100000
            taille = save_checkpoint(checkpoint_path, emb_layer, pos_embedding, transformer, fc, optimizer=optimizer, meta=meta)  # Ecrit le checkpoint final de chunk (ecriture atomique)
            print(f"checkpoint sauvegarde: {checkpoint_path} ({taille} octets)")  # Confirme la sauvegarde ET la taille reelle sur disque. Exemple: checkpoint sauvegarde: modelV3.0.pt (5243180 octets)

    return total_loss / max(total_batches, 1)  # Renvoie la perte moyenne. Exemple: total 30 / 10 batches = 3.0


if __name__ == "__main__":
    print("debut")
    print("nouveau datset")
    print("nouveau model 3.1")
    print("transformeurs corrigé")  # Ce bloc s'execute seulement si on lance ce fichier directement. Exemple: python modele_streaming.py
    parser = argparse.ArgumentParser(description="Entrainement streaming (RAM-friendly) sur dataset_ids.txt")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Nombre d'epochs (ex: 1, 5, 30)")
    args = parser.parse_args()
    epochs = args.epochs
  # Message pour voir que le script a bien demarre. Exemple affiche: debut
    context_size = CONTEXT_SIZE  # Copie la constante dans une variable locale. Exemple: 3
    emb_dim = EMB_DIM  # Taille embedding locale. Exemple: 64
    hidden_dim = HIDDEN_DIM  # Taille interne feed-forward du Transformer. Exemple: 128

    # Lecture minimale pour verifier qu'il y a assez d'IDs. Exemple: si context_size=3, il faut au moins 4 IDs.
    DATASET_IDS_PATH, ids_sample = choisir_dataset_ids_path(  # Trouve un dataset qui contient assez d'IDs. Exemple: dataset-i puis dataset_ids
        DATASET_IDS_PATH,
        fallback_path=DATASET_IDS_FALLBACK_PATH,
        min_ids=context_size + 1,
    )
    print(f"dataset utilise: {DATASET_IDS_PATH}")  # Confirme le fichier qui sera lu pendant l'entrainement. Exemple: dataset_ids.txt

    # Grace a la compression, la taille vocabulaire est bornee. Exemple: max 10000 classes en sortie.
    vocab_size = VOCAB_LIMIT  # Taille de vocabulaire utilisee par embedding et couche finale. Exemple: 10000
    id_to_token = None  # Valeur par defaut: pas de decodage texte tant qu'on n'a pas charge le vocabulaire. Exemple: None
    if PRINT_TRAINING_DECODE_EXAMPLE or PRINT_PREDICTED_TEXT:  # Charge le vocab si on veut afficher des mots pendant ou apres l'entrainement. Exemple: au moins un affichage texte demande
        id_to_token, max_raw_id, collisions = charger_vocab_pour_ids_compresses(VOCAB_JSON_PATH, vocab_limit=vocab_size)  # Construit la table id_compresse -> mot. Exemple: id_to_token[123] = "bonjour"
        if id_to_token is None:  # Si le fichier vocab.json n'existe pas ou n'est pas lisible. Exemple: vocab absent
            print(f"vocab introuvable: {VOCAB_JSON_PATH} (decodage texte desactive)")  # Informe clairement que les mots ne pourront pas etre affiches. Exemple: fallback sur IDs
        elif max_raw_id >= vocab_size or collisions > 0:  # Si des IDs bruts depassent la taille limitee, le modulo peut melanger plusieurs mots. Exemple: collisions > 0
            print("attention: comme on compresse les IDs avec %, les mots affiches peuvent etre approximatifs")  # Petit avertissement utile pour interpreter le texte affiche. Exemple: decode approximatif

    emb_layer = nn.Embedding(vocab_size, emb_dim)  # Cree la couche embedding. Exemple: [ID] -> vecteur 64
    pos_embedding = nn.Embedding(context_size, emb_dim)
    transformer_layer = nn.TransformerEncoderLayer(
        d_model=emb_dim,
        nhead=4,
        dim_feedforward=hidden_dim,
        batch_first=True,
    )
    transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
    fc = nn.Linear(emb_dim, vocab_size)  # Cree la couche de classification finale. Exemple: [128] -> [10000 scores]

    criterion = nn.CrossEntropyLoss()  # Fonction de perte pour classification multi-classes. Exemple: compare logits et ID cible
    # SGD utilise beaucoup moins de RAM que Adam. Exemple: utile sur machine limitee.
    optimizer = torch.optim.Adam(
    list(emb_layer.parameters()) + list(pos_embedding.parameters()) + list(transformer.parameters()) + list(fc.parameters()),
    lr=0.001,
    ) # Fin creation optimiseur. Exemple: pret pour optimizer.step()

    checkpoint_load_path = CHECKPOINT_LOAD_PATH  # Chemin du checkpoint a charger. Exemple: "/content/drive/MyDrive/modele_streaming.pt"
    # Detecte le device: GPU seulement si PyTorch peut vraiment executer des kernels dessus.
    device = choisir_device_compatible()
    afficher_infos_device(device)
    if RESUME_IF_CHECKPOINT_EXISTS and os.path.exists(checkpoint_load_path):  # Si on veut reprendre et que le fichier existe. Exemple: True + fichier present
        try:  # On tente de charger (tailles differentes/corruption peuvent echouer). Exemple: try/except
            # Charge le checkpoint directement sur le device detecte (evite des copies supplementaires)
            ckpt = torch.load(checkpoint_load_path, map_location=device)
            print("essai de chargement")
            # Si tu changes VOCAB_LIMIT / EMB_DIM / HIDDEN_DIM, le checkpoint n'est plus compatible.
            meta = ckpt.get("meta")  # Les tailles sauvegardees. Exemple: {"vocab_size": 10000, ...}
            if isinstance(meta, dict):  # Verifie que meta est bien un dict. Exemple: ok
                expected = {  # Ta config actuelle.
                    "vocab_size": vocab_size,
                    "emb_dim": emb_dim,
                    "hidden_dim": hidden_dim,
                    "context_size": context_size,
                }
                mismatches = []  # Stocke les differences.
                for k, v in expected.items():  # Compare chaque taille.
                    if meta.get(k) != v:
                        mismatches.append(f"{k}: ckpt={meta.get(k)} vs actuel={v}")
                if mismatches:  # Si au moins une taille differe, on refuse de charger.
                    raise ValueError("checkpoint incompatible (tailles differentes). " + " | ".join(mismatches))
            else:
                # Ancien checkpoint: pas de meta -> on compare au moins Embedding et Linear pour avoir une erreur claire.
                emb_w = ckpt.get("emb", {}).get("weight")
                fc_w = ckpt.get("fc", {}).get("weight")
                if emb_w is not None and tuple(emb_w.shape) != tuple(emb_layer.weight.shape):
                    raise ValueError(f"checkpoint incompatible (embedding): ckpt={tuple(emb_w.shape)} vs actuel={tuple(emb_layer.weight.shape)}")
                if fc_w is not None and tuple(fc_w.shape) != tuple(fc.weight.shape):
                    raise ValueError(f"checkpoint incompatible (fc): ckpt={tuple(fc_w.shape)} vs actuel={tuple(fc.weight.shape)}")

            emb_layer.load_state_dict(ckpt["emb"]) # Recharge les poids de l'embedding. Exemple: reprend l'apprentissage
            pos_embedding.load_state_dict(ckpt["pos_embedding"])  
            transformer.load_state_dict(ckpt["transformer"])  # Recharge les poids du Transformer. Exemple: reprend l'apprentissage
            fc.load_state_dict(ckpt["fc"])  # Recharge les poids de la couche finale. Exemple: reprend l'apprentissage

            # Deplacer les modules sur le device detecte (GPU si disponible)
            emb_layer.to(device)
            pos_embedding.to(device)
            transformer.to(device)
            fc.to(device)

            if "optimizer" in ckpt:  # Si l'optimiseur a ete sauvegarde aussi. Exemple: nouveau format de checkpoint
                opt_state = ckpt["optimizer"]
                opt_groups = opt_state.get("param_groups", []) if isinstance(opt_state, dict) else []
                if isinstance(optimizer, torch.optim.Adam) and any("betas" not in group for group in opt_groups):
                    print("optimizer checkpoint incompatible avec Adam: reprise avec un optimiseur neuf")
                else:
                    optimizer.load_state_dict(opt_state)  # Recharge l'etat optimiseur. Exemple: resume plus fidele
                # S'assurer que les tensors dans l'etat de l'optimiseur sont sur le bon device
                for state in optimizer.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            print(f"checkpoint charge: {checkpoint_load_path}")  # Confirme la reprise. Exemple: checkpoint charge: /content/drive/MyDrive/modele_streaming.pt
        except Exception as e:  # Si ca rate, on continue depuis zero. Exemple: tailles incompatibles
            print("impossible de charger le checkpoint, entrainement depuis zero. erreur:", e)  # Message simple. Exemple: erreur de taille
            print("astuce: garde les memes valeurs (VOCAB_LIMIT/EMB_DIM/HIDDEN_DIM) ou mets RESUME_IF_CHECKPOINT_EXISTS=False")  # Guide simple

    emb_layer.to(device)
    pos_embedding.to(device)
    transformer.to(device)
    fc.to(device)
    verifier_modules_sur_device(device, emb_layer, pos_embedding, transformer, fc)

    checkpoint_meta = build_checkpoint_meta(  # Prepare les infos communes pour toutes les sauvegardes. Exemple: tailles du modele
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        context_size=context_size,
    )

    for epoch in range(epochs):  # Boucle sur le nombre d'epochs. Exemple: --epochs 5 -> 5 iterations
        print(f"debut epoch {epoch + 1}")  # Affiche le debut d'epoch. Exemple: debut epoch 1
        loss_moyenne = train_streaming(  # Lance l'entrainement streaming pour une epoch. Exemple: retourne perte moyenne
            DATASET_IDS_PATH,  # Fichier source des IDs. Exemple: "/content/drive/MyDrive/dataset_ids.txt"
            emb_layer,  # Couche embedding a entrainer. Exemple: emb_layer
            pos_embedding,  # Embeddings de position. Exemple: pos_embedding
            transformer,  # Transformer a entrainer. Exemple: transformer
            fc,  # Couche finale a entrainer. Exemple: fc
            optimizer,  # Optimiseur pour la mise a jour des poids. Exemple: SGD
            criterion,  # Fonction de perte. Exemple: CrossEntropyLoss
            id_to_token=id_to_token,  # Donne la table de decodage au train pour pouvoir afficher les mots. Exemple: id_to_token
            context_size=context_size,  # Taille contexte utilisee. Exemple: 3
            batch_size=BATCH_SIZE,  # Taille lot utilisee. Exemple: 2
            chunk_tokens=CHUNK_TOKENS,  # Taille chunk utilisee. Exemple: 5000
            max_tokens_per_epoch=MAX_TOKENS_PER_EPOCH,  # Limite de tokens lus par epoch. Exemple: 100000
            print_decoded_example=PRINT_TRAINING_DECODE_EXAMPLE,  # Active l'affichage d'un petit exemple decode pendant l'entrainement. Exemple: True
            decoded_examples_per_chunk=TRAINING_DECODE_EXAMPLES_PER_CHUNK,  # Nombre d'exemples affiches par chunk. Exemple: 1
            checkpoint_path=CHECKPOINT_SAVE_PATH,  # Sauvegarde le modele apres chaque chunk. Exemple: modele_streaming.pt
            checkpoint_meta={**checkpoint_meta, "epoch": epoch + 1},  # Ajoute l'epoch courante dans chaque checkpoint. Exemple: epoch 1
        )  # Fin appel train_streaming. Exemple: loss_moyenne recue
        print(f"epoch {epoch + 1} loss:", loss_moyenne)  # Affiche la perte moyenne de l'epoch. Exemple: epoch 1 loss: 3.12
        print("fin 1")

    taille_finale = save_checkpoint(  # Sauvegarde finale du modele a la fin de toutes les epochs. Exemple: dernier etat complet
        CHECKPOINT_SAVE_PATH,
        emb_layer,
        pos_embedding,
        transformer,
        fc,
        optimizer=optimizer,
        meta=build_checkpoint_meta(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            context_size=context_size,
            epoch=epochs,
        ),
    )
    print(f"checkpoint final sauvegarde: {CHECKPOINT_SAVE_PATH} ({taille_finale} octets)")  # Confirme la taille reelle du checkpoint final. Exemple: checkpoint final sauvegarde: modelV3.0.pt (5243180 octets)

    emb_layer.eval()
    pos_embedding.eval()
    transformer.eval()
    fc.eval()

    x_ids = torch.tensor([ids_sample[:context_size]], dtype=torch.long, device=device)  # Construit une entree de test avec le contexte sample. Exemple: [[10,11,12]]
    with torch.no_grad():
        positions = torch.arange(x_ids.size(1), device=device).unsqueeze(0)
        vecs = embedding(x_ids, emb_layer) + pos_embedding(positions)
        out = transformer(vecs)
        logits = fc(out[:, -1, :])
        pred_id = torch.argmax(logits, dim=-1).item()

    print("vecs shape:", vecs.shape)  # Affiche la taille des embeddings de test. Exemple: torch.Size([1, 3, 64])
    print("transformer out shape:", out.shape)  # Affiche la taille de sortie Transformer de test. Exemple: torch.Size([1, 3, 128])
    print("pred_id:", pred_id)  # Affiche l'ID final predit. Exemple: pred_id: 452

    if PRINT_PREDICTED_TEXT:  # Si on veut afficher une phrase lisible. Exemple: True
        id_to_token, max_raw_id, collisions = charger_vocab_pour_ids_compresses(VOCAB_JSON_PATH, vocab_limit=vocab_size)  # Charge vocab.json et prepare table
        if id_to_token is None:  # Si vocab introuvable. Exemple: fichier absent
            print(f"vocab introuvable: {VOCAB_JSON_PATH} (affichage en IDs uniquement)")  # Message clair. Exemple: vocab introuvable
        else:  # Si vocab charge ok. Exemple: id_to_token pret
            if max_raw_id >= vocab_size or collisions > 0:  # Si vocab depasse vocab_limit, modulo peut melanger des mots. Exemple: collisions
                print("attention: comme on compresse les IDs avec %, les mots affiches peuvent etre approximatifs")  # Explication simple

        ids_prompt = load_ids(DATASET_IDS_PATH, max_ids=max(PROMPT_TOKENS, context_size))  # Charge un petit debut du fichier pour afficher. Exemple: 20 tokens
        if len(ids_prompt) < context_size:  # Si pas assez de tokens pour predire. Exemple: fichier trop court
            ids_prompt = ids_sample[:]  # Fallback: utilise l'echantillon deja charge. Exemple: au moins context_size+1 normalement

        prompt_ids = ids_prompt[:PROMPT_TOKENS]  # Tokens de depart affiches. Exemple: 20 tokens
        context_ids = ids_prompt[-context_size:]  # Contexte utilise par le modele (taille fixe). Exemple: 8 derniers tokens

        generated_ids = []  # Liste des tokens predits. Exemple: [452, 12, 98, ...]
        emb_layer.eval()  # Mode evaluation. Exemple: pas de dropout (si un jour on en ajoute)
        pos_embedding.eval()  # Mode evaluation. Exemple: stable pour generation
        transformer.eval()  # Mode evaluation. Exemple: stable pour generation
        fc.eval()  # Mode evaluation. Exemple: stable pour generation
        with torch.no_grad():  # Pas de gradients pendant la generation. Exemple: plus rapide et moins RAM
            for _ in range(GENERATE_TOKENS):  # Genere plusieurs tokens. Exemple: 20
                x_ids = torch.tensor([context_ids], dtype=torch.long, device=device)  # Cree le batch 1. Exemple: [[10,11,12,13,14,15,16,17]]
                positions = torch.arange(x_ids.size(1), device=device).unsqueeze(0)
                vecs = embedding(x_ids, emb_layer) + pos_embedding(positions) # IDs -> vecteurs. Exemple: [1,8] -> [1,8,64]
                out = transformer(vecs)  # Transformer. Exemple: [1,8,64] -> [1,8,128]
                logits = fc(out[:, -1, :])  # Scores vocab. Exemple: [1,128] -> [1,10000]
                next_id = int(torch.argmax(logits, dim=-1).item())  # Choisit l'ID le plus probable. Exemple: 452
                generated_ids.append(next_id)  # Ajoute a la liste predite. Exemple: +452
                context_ids = (context_ids + [next_id])[-context_size:]  # Glisse la fenetre de contexte. Exemple: garde 8 derniers

        phrase_depart = ids_vers_phrase(prompt_ids, id_to_token)  # Convertit le depart en mots (ou IDs). Exemple: "les bains ..."
        phrase_complete = ids_vers_phrase(prompt_ids + generated_ids, id_to_token)  # Depart + mots predits. Exemple: phrase plus longue
        print("phrase depart:", phrase_depart)  # Affiche la phrase de depart. Exemple: phrase depart: les bains ...
        print("phrase avec prediction:", phrase_complete)  # Affiche depart + predictions. Exemple: phrase avec prediction: ...

# ----------------------------------------------------------------------
# DESCRIPTION COMPLETE ET SIMPLE DU SYSTEME (SANS VOCABULAIRE COMPLEXE)
# ----------------------------------------------------------------------
# 1) Les ENTREES (inputs)
# - Le systeme lit le fichier "/content/drive/MyDrive/dataset_ids.txt".
# - Ce fichier contient une suite de nombres (des IDs de tokens), separes par des espaces.
# - Exemple d'entree du fichier: "10 11 12 13 14 15".
# - (Optionnel) Il lit aussi "vocab.json" pour afficher les IDs en mots dans la console.
#
# 2) Comment les donnees avancent dans le systeme
# - Etape A: iter_ids() lit le fichier petit morceau par petit morceau.
#   Exemple: il lit 1 bloc de texte, puis un autre bloc, au lieu de charger tout le fichier d'un coup.
# - Etape B: chaque ID est compresse avec compress_id().
#   Exemple: si VOCAB_LIMIT=10000, alors 12345 devient 2345.
# - Etape C: creer_exemples() fabrique des paires apprentissage:
#   entree X = contexte de 3 IDs, sortie y = ID suivant.
#   Exemple: [10,11,12,13] devient X=[10,11,12] et y=13.
# - Etape D: embedding transforme chaque ID en vecteur de nombres.
# - Etape E: le Transformer lit la sequence de vecteurs.
# - Etape F: couche fc transforme la sortie Transformer en scores pour tous les IDs possibles.
# - Etape G: CrossEntropyLoss compare la prediction et la vraie cible.
# - Etape H: optimizer (SGD) corrige les poids pour faire mieux au batch suivant.
#
# 3) Les SORTIES (outputs)
# - Sortie pendant l'entrainement:
#   - des messages console (chunk traite, loss d'epoch).
# - Sortie modele:
#   - fichier "/content/drive/MyDrive/modele_streaming.pt" enregistre sur disque.
#   - il contient les poids de emb, transformer et fc.
# - Sortie test final:
#   - "pred_id" affiche dans la console (ID predit pour un petit contexte de test).
#   - (Optionnel) "phrase depart" et "phrase avec prediction" affichees si vocab.json est present.
#   - tailles affichees: vecs shape et transformer out shape.
#
# 4) Ou vont les inputs et outputs (version tres directe)
# - INPUT principal: "/content/drive/MyDrive/dataset_ids.txt" -> lu par iter_ids() -> entre dans train_streaming().
# - INPUT intermediaire: X_batch/y_batch -> passe dans embedding, puis Transformer, puis fc.
# - OUTPUT intermediaire: logits, loss, pred_ids (utilises pour entrainer).
# - OUTPUT final 1: poids appris -> sauvegardes dans "/content/drive/MyDrive/modele_streaming.pt".
# - OUTPUT final 2: pred_id affiche a l'ecran pour un exemple rapide.
#
# 5) Resume en une phrase
# - Le script lit des IDs, apprend a predire le prochain ID, puis sauvegarde le modele entraine.
