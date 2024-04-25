import os
import uuid
import numpy as np
from fastapi import FastAPI, File, UploadFile
from redis.commands.search.query import Query

from app.redis_db import r
from app.utils import represent_image_model

app = FastAPI()


@app.get("/")
async def root(image: UploadFile = File(...)):
    img_extension = os.path.splitext(image.filename)[1]
    target_img = f"app/images/{uuid.uuid4().hex}{img_extension}"

    # Save the uploaded image to a temporary file
    with open(target_img, "wb") as f:
        f.write(await image.read())

    target_embedding = represent_image_model(target_img)
    os.remove(target_img)
    query_vector = np.array(target_embedding).astype(np.float32).tobytes()
    k = 8
    base_query = f"*=>[KNN {k} @embedding $query_vector AS distance]"
    query = Query(base_query).return_fields("distance").sort_by("distance").dialect(2)
    results = r.ft().search(query, query_params={"query_vector": query_vector})
    for i in results.docs:
        print(i.id, "=>", round(100 - (float(i.distance) * 100), 2), "%")
    return results.docs


@app.post("/sorting")
async def sorting_images(images: list[UploadFile] = File(...)):
    embeddings = []

    for image in images:
        # Generate a random filename for the image
        img_extension = os.path.splitext(image.filename)[1]
        img_path = f"app/images/{uuid.uuid4().hex}{img_extension}"

        # Save the uploaded image to a temporary file
        with open(img_path, "wb") as f:
            f.write(await image.read())

        # Generate the embedding using DeepFace
        embedding = represent_image_model(img_path)
        embeddings.append((img_path, embedding))
        # Clean up the temporary file
        os.remove(img_path)

    pipeline = r.pipeline(transaction=False)
    for img_path, embedding in embeddings:
        key = img_path.split("/")[-1]
        value = np.array(embedding).astype(np.float32).tobytes()

        pipeline.hset(key, mapping={"embedding": value, "NAME": key})
        pipeline.execute()

    return {"message": "Images processed successfully"}
