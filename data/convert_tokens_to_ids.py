import argparse
import json
import os
import sqlite3
import time
from collections import OrderedDict
from json import JSONDecodeError


class LruCache:
    def __init__(self, max_size):
        self.max_size = max(0, int(max_size))
        self.data = OrderedDict()

    def get(self, key):
        if self.max_size == 0:
            return None
        value = self.data.get(key)
        if value is not None:
            self.data.move_to_end(key)
        return value

    def put(self, key, value):
        if self.max_size == 0:
            return
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.max_size:
            self.data.popitem(last=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convertit un dataset de tokens en IDs sans charger tout le dataset ni tout le vocabulaire en RAM."
    )
    parser.add_argument(
        "--input",
        default=os.path.join("dataset", "dataset_token.txt"),
        help="Fichier texte contenant les tokens separes par des espaces ou retours ligne.",
    )
    parser.add_argument(
        "--output",
        default="dataset_ids.txt",
        help="Fichier de sortie contenant les IDs separes par des espaces.",
    )
    parser.add_argument(
        "--vocab-json",
        default="vocab.json",
        help="Fichier vocabulaire JSON au format {token: id}.",
    )
    parser.add_argument(
        "--vocab-db",
        default="vocab.sqlite3",
        help="Index SQLite utilise pour travailler sans charger tout le vocabulaire en RAM.",
    )
    parser.add_argument(
        "--read-chars",
        type=int,
        default=4_000_000,
        help="Taille de lecture par bloc de texte.",
    )
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=120_000,
        help="Nombre de tokens a traiter avant une ecriture disque.",
    )
    parser.add_argument(
        "--lookup-batch",
        type=int,
        default=900,
        help="Nombre max de tokens uniques recherches en une requete SQLite.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=200_000,
        help="Taille du cache memoire token->id pour accelerer les tokens frequents.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=2_000_000,
        help="Affiche une ligne de progression tous les N tokens.",
    )
    parser.add_argument(
        "--force-rebuild-db",
        action="store_true",
        help="Reconstruit l'index SQLite a partir de vocab.json meme si vocab.sqlite3 existe deja.",
    )
    return parser.parse_args()


def configure_sqlite(conn):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-20000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vocab (token TEXT PRIMARY KEY, id INTEGER NOT NULL)"
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_vocab_id ON vocab(id)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.commit()


def skip_ws(buffer, pos):
    length = len(buffer)
    while pos < length and buffer[pos].isspace():
        pos += 1
    return pos


def stream_flat_json_object(path, chunk_size=1_000_000):
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        pos = 0
        started = False
        finished = False
        eof = False

        while not finished:
            if pos > 0 and (pos > chunk_size or eof):
                buffer = buffer[pos:]
                pos = 0

            if not eof and len(buffer) - pos < chunk_size // 2:
                morceau = f.read(chunk_size)
                if morceau:
                    buffer += morceau
                else:
                    eof = True

            pos = skip_ws(buffer, pos)

            if not started:
                if pos >= len(buffer):
                    if eof:
                        return
                    continue
                if buffer[pos] != "{":
                    raise ValueError(f"{path} ne contient pas un objet JSON a la racine.")
                pos += 1
                started = True
                continue

            pos = skip_ws(buffer, pos)
            if pos >= len(buffer):
                if eof:
                    raise ValueError(f"{path} est incomplet.")
                continue

            if buffer[pos] == "}":
                finished = True
                break

            while True:
                try:
                    key, next_pos = decoder.raw_decode(buffer, pos)
                    break
                except JSONDecodeError:
                    if eof:
                        raise
                    morceau = f.read(chunk_size)
                    if not morceau:
                        eof = True
                    else:
                        buffer += morceau

            pos = skip_ws(buffer, next_pos)
            while pos >= len(buffer) and not eof:
                morceau = f.read(chunk_size)
                if not morceau:
                    eof = True
                else:
                    buffer += morceau

            if pos >= len(buffer) or buffer[pos] != ":":
                raise ValueError(f"Separateur ':' manquant dans {path}.")
            pos += 1

            while True:
                pos = skip_ws(buffer, pos)
                try:
                    value, pos = decoder.raw_decode(buffer, pos)
                    break
                except JSONDecodeError:
                    if eof:
                        raise
                    morceau = f.read(chunk_size)
                    if not morceau:
                        eof = True
                    else:
                        buffer += morceau

            yield key, value

            while True:
                pos = skip_ws(buffer, pos)
                if pos < len(buffer):
                    break
                if eof:
                    raise ValueError(f"{path} est incomplet apres une entree.")
                morceau = f.read(chunk_size)
                if not morceau:
                    eof = True
                else:
                    buffer += morceau

            if buffer[pos] == ",":
                pos += 1
                continue
            if buffer[pos] == "}":
                finished = True
                break
            raise ValueError(f"Separateur inattendu dans {path}: {buffer[pos]!r}")


def import_vocab_json_to_db(vocab_json_path, vocab_db_path):
    if not os.path.exists(vocab_json_path):
        raise FileNotFoundError(f"Fichier introuvable: {vocab_json_path}")

    temp_db_path = vocab_db_path + ".tmp"
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

    conn = sqlite3.connect(temp_db_path)
    try:
        configure_sqlite(conn)
        insert_sql = "INSERT INTO vocab(token, id) VALUES (?, ?)"
        count = 0
        max_id = -1
        batch = []

        for token, token_id in stream_flat_json_object(vocab_json_path):
            token_id = int(token_id)
            batch.append((str(token), token_id))
            if token_id > max_id:
                max_id = token_id
            if len(batch) >= 50_000:
                conn.executemany(insert_sql, batch)
                conn.commit()
                count += len(batch)
                print(f"[import vocab] {count:,} entrees indexees...")
                batch.clear()

        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()
            count += len(batch)
            print(f"[import vocab] {count:,} entrees indexees...")

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('next_id', ?)",
            (str(max_id + 1),),
        )
        conn.commit()
    finally:
        conn.close()

    if os.path.exists(vocab_db_path):
        os.remove(vocab_db_path)
    os.replace(temp_db_path, vocab_db_path)


def ensure_vocab_db(vocab_json_path, vocab_db_path, force_rebuild=False):
    rebuild = force_rebuild

    if not os.path.exists(vocab_db_path):
        rebuild = True
    elif os.path.exists(vocab_json_path):
        rebuild = os.path.getmtime(vocab_json_path) > os.path.getmtime(vocab_db_path)

    if rebuild:
        print("[setup] construction de l'index SQLite a partir de vocab.json...")
        import_vocab_json_to_db(vocab_json_path, vocab_db_path)
    else:
        print("[setup] reutilisation de l'index SQLite existant.")


def get_next_id(conn):
    row = conn.execute("SELECT value FROM meta WHERE key='next_id'").fetchone()
    if row is not None:
        return int(row[0])
    row = conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM vocab").fetchone()
    return int(row[0])


def iter_token_batches(path, read_chars, batch_tokens):
    with open(path, "r", encoding="utf-8") as f:
        pending = []
        remainder = ""

        while True:
            chunk = f.read(read_chars)
            if not chunk:
                break

            text = remainder + chunk
            parts = text.split()

            if text and not text[-1].isspace():
                remainder = parts.pop() if parts else text
            else:
                remainder = ""

            if parts:
                pending.extend(parts)

            if len(pending) >= batch_tokens:
                yield pending, f.buffer.tell()
                pending = []

        if remainder:
            pending.append(remainder)

        if pending:
            yield pending, f.buffer.tell()


def fetch_existing_ids(conn, tokens, lookup_batch):
    found = {}
    for start in range(0, len(tokens), lookup_batch):
        subset = tokens[start : start + lookup_batch]
        placeholders = ",".join("?" for _ in subset)
        query = f"SELECT token, id FROM vocab WHERE token IN ({placeholders})"
        for token, token_id in conn.execute(query, subset):
            found[token] = int(token_id)
    return found


def export_vocab_db_to_json(conn, vocab_json_path):
    temp_path = vocab_json_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        first = True
        for token, token_id in conn.execute("SELECT token, id FROM vocab ORDER BY id"):
            if not first:
                f.write(",\n")
            f.write(f"  {json.dumps(token, ensure_ascii=False)}: {int(token_id)}")
            first = False
        f.write("\n}\n")
    os.replace(temp_path, vocab_json_path)


def convert_dataset(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Dataset introuvable: {args.input}")

    ensure_vocab_db(args.vocab_json, args.vocab_db, args.force_rebuild_db)

    total_bytes = os.path.getsize(args.input)
    cache = LruCache(args.cache_size)
    start_time = time.time()
    total_tokens = 0
    new_tokens = 0

    conn = sqlite3.connect(args.vocab_db)
    try:
        configure_sqlite(conn)
        next_id = get_next_id(conn)

        with open(args.output, "w", encoding="utf-8") as f_out:
            for tokens, bytes_read in iter_token_batches(
                args.input, args.read_chars, args.batch_tokens
            ):
                missing = []
                missing_seen = set()

                for token in tokens:
                    token_id = cache.get(token)
                    if token_id is None and token not in missing_seen:
                        missing.append(token)
                        missing_seen.add(token)

                existing = fetch_existing_ids(conn, missing, args.lookup_batch)
                resolved_ids = dict(existing)

                inserts = []
                for token in missing:
                    token_id = existing.get(token)
                    if token_id is None:
                        token_id = next_id
                        next_id += 1
                        new_tokens += 1
                        inserts.append((token, token_id))
                    resolved_ids[token] = token_id
                    cache.put(token, token_id)

                if inserts:
                    conn.executemany("INSERT INTO vocab(token, id) VALUES (?, ?)", inserts)
                    conn.commit()

                ids_as_text = []
                for token in tokens:
                    token_id = cache.get(token)
                    if token_id is None:
                        token_id = resolved_ids[token]
                        cache.put(token, token_id)
                    ids_as_text.append(str(token_id))

                f_out.write(" ".join(ids_as_text))
                f_out.write("\n")

                total_tokens += len(tokens)

                if (
                    args.report_every > 0
                    and total_tokens
                    and total_tokens % args.report_every < len(tokens)
                ):
                    elapsed = max(time.time() - start_time, 1e-9)
                    rate = total_tokens / elapsed
                    pct = (bytes_read / total_bytes * 100.0) if total_bytes else 0.0
                    print(
                        "[progress] "
                        f"{total_tokens:,} tokens | "
                        f"{new_tokens:,} nouveaux | "
                        f"{rate:,.0f} tok/s | "
                        f"{pct:5.1f}% lu"
                    )

            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('next_id', ?)",
                (str(next_id),),
            )
            conn.commit()

        print("[export] reecriture de vocab.json...")
        export_vocab_db_to_json(conn, args.vocab_json)
    finally:
        conn.close()

    elapsed = time.time() - start_time
    print(
        "[termine] "
        f"{total_tokens:,} tokens convertis, "
        f"{new_tokens:,} nouveaux tokens ajoutes au vocabulaire, "
        f"{elapsed:.1f}s"
    )


def main():
    args = parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()
