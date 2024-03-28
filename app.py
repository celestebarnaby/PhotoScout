from flask import Flask, request, jsonify
import json
from utils import *
from synthesizer import *
import torch
from user_study_tasks import tasks
from gpt import *

from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
img_to_environment = {}
img_to_embedding = {}
logged_info = {}
args = get_args()

with open("gpt_cache.json") as f:
    cache = json.load(f)

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the model ID
model_ID = "openai/clip-vit-base-patch32"
# Get model, processor & tokenizer
model, processor, tokenizer = get_model_info(model_ID, device)


@app.route("/tagImage", methods=["POST"])
@cross_origin()
def tag_image():
    global img_to_environment
    global logged_info
    global img_to_embedding

    body = request.get_json()
    tags = body["tags"]
    for env in img_to_environment.values():
        env = env["environment"]
        for obj in env.values():
            if "Index" not in obj:
                continue
            if str(obj["Index"]) in tags:
                obj["Tag"] = tags[str(obj["Index"])]["text"]
            elif "Tag" in obj:
                del obj["Tag"]
            obj["Description"] = get_description(obj, tags)
    return {"message": img_to_environment}


@app.route("/logSavedImages", methods=["POST"])
@cross_origin()
def log_saved_images():
    global logged_info

    return {"success": True}


@app.route("/textQuery", methods=["POST"])
@cross_origin()
def text_query():
    global img_to_environment
    global logged_info
    global img_to_embedding
    global cache

    args = get_args()
    body = request.get_json()
    text_query = body["text_query"]
    examples = body["examples"]
    tags = body["tags"]
    tags = set([tag["text"].lower() for tag in tags.values()])
    logged_info["text queries"].append(text_query)
    logged_info["example images"].append(examples)
    start_time = time.perf_counter()
    try:
        results, robot_text, robot_text2, prog = make_text_query(
            text_query,
            img_to_environment,
            list(examples.items()),
            tags,
            cache,
            args.use_cache,
        )
    except TimeoutError:
        results = []
        robot_text = """
Your query timed out. Try changing your text query, or removing some example images.
"""
        robot_text2 = ""
        prog = None
    logged_info["synthesized_progs"].append(prog)
    logged_info["synthesis_results"].append(
        (results, time.perf_counter() - logged_info["start time"])
    )
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print("TIME: {}".format(total_time))
    return {
        "search_results": results,
        "robot_text": robot_text,
        "robot_text2": robot_text2,
    }


@app.route("/loadFiles", methods=["POST"])
@cross_origin()
def load_files():
    global img_to_environment
    global img_to_embedding
    global logged_info
    task_num = request.get_json()
    task = tasks[task_num]
    img_to_embedding = {}
    img_folder = "photoscout_ui/public/images/" + task["dataset"] + "/"
    img_to_environment = preprocess(img_folder, 100)
    logged_info["task"] = task["description"]
    logged_info["dataset"] = task["dataset"]
    logged_info["text queries"] = []
    logged_info["example images"] = []
    logged_info["num"] = task_num
    logged_info["start time"] = time.perf_counter()
    logged_info["synthesis_results"] = []
    logged_info["synthesized_progs"] = []
    logged_info["f1_score_of_saved_images"] = []
    return {
        "message": img_to_environment,
        "files": [filename for filename in img_to_environment.keys()],
        "task_description": task["description"],
    }


@app.route("/submitResults", methods=["POST"])
@cross_origin()
def log_results():
    global img_to_environment
    global img_to_embedding
    response = {
        "status": "ok",
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="localhost", port=5001, debug=True)
