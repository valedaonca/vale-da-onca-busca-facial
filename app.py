"""
Busca de fotos por reconhecimento facial — Vale da Onca
=========================================================
Versão Streamlit — roda no Streamlit Community Cloud (gratuito, sem cartão).

Duas páginas, escolhidas no menu lateral:
  1) "Buscar minhas fotos" (uso do participante) — envia uma selfie e
     recebe as fotos do álbum que têm esse rosto.
  2) "Indexar fotos (equipe)" — lê todas as fotos de um álbum do Flickr,
     detecta rostos e guarda a "assinatura" de cada um.

Modelo usado: ArcFace (biblioteca DeepFace), testado e validado como o mais
confiável entre as opções gratuitas — um modelo mais simples (Facenet)
chegou a confundir duas pessoas diferentes nos nossos testes.
"""

import os
import pickle
import time

import numpy as np
import requests
import streamlit as st
from deepface import DeepFace

# ============================================================
# CONFIG — edite pra cada evento novo
# ============================================================

FLICKR_USER_ID_DEFAULT = "199195039@N05"
FLICKR_PHOTOSET_ID_DEFAULT = "72177720334552032"

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "mtcnn"
DISTANCE_THRESHOLD = 0.68  # quanto MENOR, mais parecido — validado nos testes

EMBEDDINGS_FILE = "embeddings.pkl"

# ============================================================

st.set_page_config(page_title="Vale da Onça — Busca por rosto", page_icon="🐆", layout="centered")


def get_all_flickr_photos(api_key, user_id, photoset_id, status_area=None):
    photos = []
    page = 1
    while True:
        resp = requests.get(
            "https://www.flickr.com/services/rest/",
            params={
                "method": "flickr.photosets.getPhotos",
                "api_key": api_key,
                "photoset_id": photoset_id,
                "user_id": user_id,
                "extras": "url_o,url_l,url_c",
                "per_page": 500,
                "page": page,
                "format": "json",
                "nojsoncallback": 1,
            },
            timeout=30,
        )
        data = resp.json()
        if data.get("stat") != "ok":
            raise RuntimeError(f"Erro na API do Flickr: {data}")

        photoset = data["photoset"]
        photos.extend(photoset["photo"])
        if status_area:
            status_area.write(f"Página {page}/{photoset['pages']} — {len(photos)} fotos até agora")

        if page >= photoset["pages"]:
            break
        page += 1

    return photos


def best_image_url(photo):
    for key in ("url_o", "url_l", "url_c"):
        if photo.get(key):
            return photo[key]
    return None


def flickr_page_url(user_id, photoset_id, photo_id):
    return f"https://www.flickr.com/photos/{user_id}/{photo_id}/in/album-{photoset_id}"


def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return []


def save_embeddings(entries):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(entries, f)


def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)


# ------------------------------------------------------------
# Página: Buscar minhas fotos
# ------------------------------------------------------------

def page_search():
    st.title("🐆 Ache suas fotos")
    st.caption("TOP 2000 DUPLO X — Vale da Onça")
    st.write(
        "Envie uma selfie de frente, com boa luz, e a gente encontra suas "
        "fotos no álbum do evento."
    )

    selfie_file = st.file_uploader("Sua selfie", type=["jpg", "jpeg", "png"])

    if selfie_file is not None:
        st.image(selfie_file, width=200)

        if st.button("Buscar minhas fotos", type="primary"):
            entries = load_embeddings()
            if not entries:
                st.warning(
                    "Ainda não há fotos indexadas. Peça pra equipe rodar "
                    "a página 'Indexar fotos' primeiro."
                )
                return

            tmp_path = "/tmp/_selfie.jpg"
            with open(tmp_path, "wb") as f:
                f.write(selfie_file.getbuffer())

            with st.spinner("Procurando..."):
                try:
                    faces = DeepFace.represent(
                        img_path=tmp_path,
                        model_name=MODEL_NAME,
                        detector_backend=DETECTOR_BACKEND,
                        enforce_detection=True,
                    )
                except Exception:
                    st.error(
                        "Não conseguimos detectar um rosto nessa foto. "
                        "Tente uma selfie de frente, com boa luz."
                    )
                    return

                if not faces:
                    st.error("Não conseguimos detectar um rosto nessa foto.")
                    return

                selfie_embedding = np.array(faces[0]["embedding"])

                best_per_photo = {}
                for e in entries:
                    dist = cosine_distance(selfie_embedding, e["embedding"])
                    if dist > DISTANCE_THRESHOLD:
                        continue
                    photo_id = e["photo_id"]
                    if photo_id not in best_per_photo or dist < best_per_photo[photo_id]["dist"]:
                        best_per_photo[photo_id] = {
                            "dist": dist,
                            "page_url": e["page_url"],
                            "image_url": e["image_url"],
                        }

                results = sorted(best_per_photo.values(), key=lambda x: x["dist"])

            if not results:
                st.info("Não encontramos fotos com esse rosto no álbum indexado.")
                return

            st.success(f"{len(results)} foto(s) encontrada(s)!")
            cols = st.columns(3)
            for i, r in enumerate(results):
                similarity = round((1 - r["dist"] / DISTANCE_THRESHOLD) * 100)
                with cols[i % 3]:
                    st.image(r["image_url"], use_container_width=True)
                    st.caption(f"{similarity}% parecido")
                    st.markdown(f"[Abrir no Flickr]({r['page_url']})")

    st.divider()
    st.caption(
        "**Privacidade:** sua selfie é usada só na hora da busca, comparada "
        "com os rostos das fotos do evento, e não fica guardada em lugar "
        "nenhum. O reconhecimento facial usa dado biométrico, tratado "
        "conforme a LGPD."
    )


# ------------------------------------------------------------
# Página: Indexar fotos (equipe)
# ------------------------------------------------------------

def page_index():
    st.title("⚙️ Indexar fotos (uso interno da equipe)")
    st.write(
        "Roda uma vez por evento, depois que todas as fotos estiverem "
        "publicadas no álbum do Flickr. Pode ser interrompido e rodado de "
        "novo sem duplicar trabalho."
    )

    flickr_api_key = st.text_input("Chave da API do Flickr", type="password")
    flickr_user_id = st.text_input("Flickr User ID", value=FLICKR_USER_ID_DEFAULT)
    flickr_photoset_id = st.text_input("Flickr Photoset ID", value=FLICKR_PHOTOSET_ID_DEFAULT)

    if st.button("Indexar fotos", type="primary"):
        if not flickr_api_key:
            st.error("Preencha a chave da API do Flickr.")
            return

        entries = load_embeddings()
        already_done = {e["photo_id"] for e in entries}

        status_area = st.empty()
        status_area.write("Buscando lista de fotos no Flickr...")
        photos = get_all_flickr_photos(flickr_api_key, flickr_user_id, flickr_photoset_id, status_area)
        total = len(photos)

        st.write(f"Total de fotos no álbum: {total} — já indexadas antes: {len(already_done)}")

        progress_bar = st.progress(0)
        log_area = st.empty()
        indexed_photos = 0
        indexed_faces = 0
        errors = 0

        for i, photo in enumerate(photos):
            progress_bar.progress((i + 1) / total)

            if photo["id"] in already_done:
                continue

            image_url = best_image_url(photo)
            if not image_url:
                continue

            try:
                img_resp = requests.get(image_url, timeout=30)
                tmp_path = "/tmp/_idx_photo.jpg"
                with open(tmp_path, "wb") as f:
                    f.write(img_resp.content)

                faces = DeepFace.represent(
                    img_path=tmp_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                )

                page_url = flickr_page_url(flickr_user_id, flickr_photoset_id, photo["id"])

                for face in faces:
                    if face.get("face_confidence", 1) <= 0:
                        continue
                    entries.append({
                        "photo_id": photo["id"],
                        "embedding": np.array(face["embedding"]),
                        "page_url": page_url,
                        "image_url": image_url,
                    })
                    indexed_faces += 1

                indexed_photos += 1

            except Exception as e:
                errors += 1
                log_area.write(f"Erro na foto {photo['id']}: {e}")

            if (i + 1) % 25 == 0:
                save_embeddings(entries)

        save_embeddings(entries)

        st.success(
            f"Concluído! Fotos processadas nesta rodada: {indexed_photos} | "
            f"Rostos indexados: {indexed_faces} | Erros: {errors} | "
            f"Total acumulado: {len(entries)} rostos"
        )

        with open(EMBEDDINGS_FILE, "rb") as f:
            st.download_button(
                "Baixar embeddings.pkl (backup)",
                data=f,
                file_name="embeddings.pkl",
            )

        st.warning(
            "IMPORTANTE: pra esse índice não se perder se o app reiniciar, "
            "baixe o arquivo acima e suba ele de volta no seu repositório "
            "do GitHub (substituindo o embeddings.pkl que já está lá, se "
            "houver), depois faça commit."
        )


# ------------------------------------------------------------
# Navegação
# ------------------------------------------------------------

page = st.sidebar.radio("Navegação", ["Buscar minhas fotos", "Indexar fotos (equipe)"])

if page == "Buscar minhas fotos":
    page_search()
else:
    page_index()
