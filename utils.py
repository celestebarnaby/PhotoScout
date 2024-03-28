from image_utils import *
import argparse
import os
import json
import csv
import time
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# from typesystem import *

DETAIL_KEYS = [
    "Eyeglasses",
    "Sunglasses",
    "Beard",
    "Mustache",
    "EyesOpen",
    "Smile",
    "MouthOpen",
    "Emotions",
    "AgeRange",
]


class TimeOutException(Exception):
    def __init__(self, message, errors=None):
        super(TimeOutException, self).__init__(message)
        self.errors = errors


class Hole:
    def __init__(self, depth, node_type, output_over=None, output_under=None, val=None):
        self.depth = depth
        self.node_type = node_type
        self.output_over = output_over
        self.output_under = output_under
        self.val = None

    def __str__(self):
        return type(self).__name__

    def duplicate(self):
        return Hole(
            self.depth, self.node_type, self.output_over, self.output_under, self.val
        )

    def __lt__(self, other):
        if not isinstance(other, Hole):
            return False
        return str(self) < str(other)


def handler(signum, frame):
    raise TimeOutException("Timeout")


def get_output_objs(env, action):
    objs = set()
    # print("env:", env)
    for obj_id, details_map in env.items():
        if "ActionApplied" in details_map and details_map["ActionApplied"] == action:
            objs.add(obj_id)
    return ",".join(sorted(objs))


def compare_objs_with_output(env, objs, action):
    match = True
    for id_ in objs:
        if "ActionApplied" not in env[id_] or env[id_]["ActionApplied"] != action:
            match = False
    for id_, details_map in env.items():
        if (
            "ActionApplied" in details_map
            and details_map["ActionApplied"] == action
            and not id_ in objs
        ):
            match = False
    return match


def get_max_scoring_image(img_to_environment):
    return max(img_to_environment, key=lambda k: img_to_environment[k]["score"])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_cache",
        type=bool,
        default=False,
        help="if True, manually inspect output program on sampled images",
    )
    args = parser.parse_args()
    return args


def get_obj_str(obj, use_index=False):
    if obj["Type"] == "Text":
        return "Text" + obj["Text"]
    if obj["Type"] == "Object":
        return "Object" + obj["Name"]
    else:
        props = "".join(
            [prop for prop in ["Smile", "EyesOpen", "MouthOpen"] if prop in obj]
        )
        if use_index:
            return "Face" + props + str(obj["Index"])
        return "Face" + props


def get_obj_strs(objs):
    strs = []
    for obj in objs:
        strs.append(get_obj_str(obj))
        strs.append(get_obj_str(obj, True))
    return strs


def get_image_embedding(image, processor, device, model):
    image = processor(text=None, images=image, return_tensors="pt")["pixel_values"].to(
        device
    )
    embedding = model.get_image_features(image)
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np


def preprocess_embeddings(img_folder, img_to_environment, processor, device, model):
    d = {}
    if os.path.exists("./embeddings.json"):
        with open("./embeddings.json", "r") as fp:
            d = json.load(fp)
            if img_folder in d:
                return d[img_folder]
    img_to_embedding = {}
    for image_name in img_to_environment:
        image = Image.open(image_name)
        img_to_embedding[image_name] = get_image_embedding(
            image, processor, device, model
        ).tolist()
    d[img_folder] = img_to_embedding
    with open("embeddings.json", "w") as fp:
        json.dump(d, fp)
    return img_to_embedding


def preprocess(img_folder, max_faces=100):
    """
    Given an img_folder, cache all the image's information to a dict, scored by the strategy
    """

    print("loading images and preprocessing...")

    # read the cache if it exists
    key = img_folder
    test_images = {}

    print(os.path.exists("./test_images_ui.json"))

    if os.path.exists("./test_images_ui.json"):
        with open("./test_images_ui.json", "r") as fp:
            test_images = json.load(fp)
            if key in test_images:
                return test_images[key]
    client = get_client()
    client.delete_collection(CollectionId="library2")
    client.create_collection(CollectionId="library2")
    img_to_environment = {}
    prev_env = {}
    img_index = 0

    start_time = time.perf_counter()
    obj_strs = set()
    # loop through all image files to cache information
    for filename in os.listdir(img_folder):
        print("filename:", filename)
        if filename.endswith("DS_Store"):
            continue
        img_dir = img_folder + filename
        env = get_environment(
            [img_dir], client, img_index, DETAIL_KEYS, prev_env, max_faces
        )
        obj_strs.update(get_obj_strs(env.values()))
        img = cv2.imread(img_dir, 1)
        height, width, _ = img.shape
        print((width, height))
        # print("environment:", env)
        score = len(env)
        # print("score:", score)
        img_to_environment[img_dir] = {
            "environment": env,
            "img_index": img_index,
            "score": score,
            "dimensions": [width, height],
        }
        if not env:
            continue
        img_index += 1
        prev_env = prev_env | env
    end_time = time.perf_counter()
    total_time = end_time - start_time

    print("preprocessing finished...")

    clean_environment(img_to_environment)
    obj_strs_sorted = sorted(list(obj_strs))

    print("Num images: ", len(os.listdir(img_folder)))
    print("Total time: ", total_time)
    add_descriptions(img_to_environment)
    test_images[key] = img_to_environment
    test_images[key + "obj_str"] = obj_strs_sorted
    # print(img_to_environment)

    with open("test_images_ui.json", "w") as fp:
        json.dump(test_images, fp)

    return img_to_environment, obj_strs_sorted


def add_descriptions(img_to_environment):
    for img, lib in img_to_environment.items():
        env = lib["environment"]
        for obj in env.values():
            obj["Description"] = get_description(obj)


def get_description(obj, tags={}):
    if obj["Type"] == "Object":
        if obj["Name"].lower() in name_to_parent:
            return "{}, {}".format(
                name_to_parent[obj["Name"].lower()], obj["Name"]
            ).capitalize()
        return obj["Name"]
    if obj["Type"] == "Text":
        return "Text that reads '{}'".format(obj["Text"])
    addl_features = []
    if "Smile" in obj:
        addl_features.append("is smiling")
    if "EyesOpen" in obj:
        addl_features.append("has eyes open")
    if "Index" in obj:
        if str(obj["Index"]) in tags:
            addl_features.append("tagged as {}".format(tags[str(obj["Index"])]["text"]))
        else:
            addl_features.append("has id {}".format(obj["Index"]))
    if addl_features:
        return "Face that {}, and is between {} and {} years old".format(
            ", ".join(addl_features), obj["AgeRange"]["Low"], obj["AgeRange"]["High"]
        )
    else:
        return "Face that is between {} and {} years old".format(
            obj["AgeRange"]["Low"], obj["AgeRange"]["High"]
        )


# Replace face hashes with readable face ids
def clean_environment(img_to_environment):
    new_id = "0"
    for lib in img_to_environment.values():
        new_env = {}
        env = lib["environment"]
        for face_hash, face_details in env.items():
            new_env[new_id] = face_details
            new_id = str(int(new_id) + 1)
        lib["environment"] = new_env


def write_logs(logs):
    total_time_per_task = {}
    for row in logs:
        task = row[:2]
        if task not in total_time_per_task:
            total_time_per_task[task] = 0
        total_time_per_task[task] += row[7]
    for task, total_time in total_time_per_task.items():
        row = (task[0] + "_TOTAL", task[1], "", "", "", "", "", total_time)
        logs.append(row)
    with open("data/logs.csv", "w") as f:
        fw = csv.writer(f)
        fw.writerow(
            (
                "Operation",
                "Task ID",
                "# Objects",
                "# States",
                "# Transitions",
                "Max CPU Usage",
                "Max Memory Usage",
                "Total Time",
            )
        )
        for row in logs:
            fw.writerow(row)


def write_synthesis_overview(logs, ablation_name):
    filename = "data/synthesis_overview" + ablation_name + ".csv"
    with open(filename, "w") as f:
        fw = csv.writer(f)
        for row in logs:
            fw.writerow(row)


def get_valid_indices(env, output_under, output_over):
    req_indices = set(
        [env[obj_id]["Index"] for obj_id in output_under if "Index" in env[obj_id]]
    )
    if len(req_indices) == 1:
        return req_indices
    if len(req_indices) > 1:
        return []
    return sorted(
        list(
            set(
                [
                    env[obj_id]["Index"]
                    for obj_id in output_over
                    if "Index" in env[obj_id]
                ]
            )
        )
    )


def get_valid_words(env, output_under, output_over):
    req_words = set(
        [
            env[obj_id]["Text"].lower()
            for obj_id in output_under
            if "Text" in env[obj_id]
        ]
    )
    if len(req_words) == 1:
        return req_words
    if len(req_words) > 1:
        return []
    return sorted(
        list(
            set(
                [
                    env[obj_id]["Text"].lower()
                    for obj_id in output_over
                    if "Text" in env[obj_id]
                ]
            )
        )
    )


def get_valid_objects(env, output_under, output_over):
    req_objects = set(
        [env[obj_id]["Name"] for obj_id in output_under if "Name" in env[obj_id]]
    )
    if len(req_objects) == 1:
        return req_objects
    if len(req_objects) > 1:
        return []
    return sorted(
        list(
            set(
                [env[obj_id]["Name"] for obj_id in output_over if "Name" in env[obj_id]]
            )
        )
    )


def get_text_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    text_embeddings = model.get_text_features(**inputs)
    # convert the embeddings to numpy array
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    return embedding_as_np


def get_image_embedding(image, processor, device, model):
    image = processor(text=None, images=image, return_tensors="pt")["pixel_values"].to(
        device
    )
    embedding = model.get_image_features(image)
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np


def get_model_info(model_ID, device):
    # Save the model to device
    model = CLIPModel.from_pretrained(model_ID).to(device)
    # Get the processor
    processor = CLIPProcessor.from_pretrained(model_ID)
    # Get the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    # Return model, processor & tokenizer
    return model, processor, tokenizer


def get_top_N_images(
    query,
    tokenizer,
    model,
    processor,
    device,
    img_to_embedding,
    top_K=4,
    search_criterion="text",
):
    image_names = img_to_embedding.keys()
    image_embeddings = [np.array(img_to_embedding[name]) for name in image_names]
    threshold = (
        0.2
        if search_criterion == "text"
        else 0.75
        if search_criterion == "image"
        else 0
    )

    # Text to image Search
    if search_criterion.lower() in {"text", "imageeye"}:
        query_vect = get_text_embedding(query, tokenizer, model)
    # Image to image Search
    else:
        query_vect = get_image_embedding(query, processor, device, model)
    # Run similarity Search
    print(type(query_vect))
    print(type(image_embeddings[0]))
    cos_sim = [cosine_similarity(query_vect, x) for x in image_embeddings]
    cos_sim = [x[0][0] for x in cos_sim]
    cos_sim_per_image = zip(cos_sim, image_names)
    most_similar = sorted(cos_sim_per_image, reverse=True)
    print(most_similar)
    # [1:top_K+1]  # line 24
    top_images = [img for (cos_sim, img) in most_similar if cos_sim > threshold]
    return top_images
