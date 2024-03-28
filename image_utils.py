import boto3
import csv
import cv2
import re
from typing import Any, List, Dict


# Since we don't construct class here, some static variables to track things
face_hash_to_id: Dict[str, int] = {}
face_id_to_hash: Dict[int, str] = {}


def add_face_hash_id_mapping(hash_code: str, face_id: int):
    global face_id_to_hash
    global face_hash_to_id

    face_id_to_hash[face_id] = hash_code
    face_hash_to_id[hash_code] = face_id


def is_contained(bbox1, bbox2, include_edges=False):
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2
    if include_edges:
        return (
            left1 >= left2 and top1 >= top2 and bottom1 <= bottom2 and right1 <= right2
        )
    else:
        return left1 > left2 and top1 > top2 and bottom1 < bottom2 and right1 < right2


def get_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    left1, top1, right1, bottom1 = bbox1
    left2, top2, right2, bottom2 = bbox2

    # determine the coordinates of the intersection rectangle
    int_left = max(left1, left2)
    int_top = max(top1, top2)
    int_right = min(right1, right2)
    int_bottom = min(bottom1, bottom2)

    if int_right < int_left or int_bottom < int_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (int_right - int_left) * (int_bottom - int_top)

    # compute the area of both AABBs
    bb1_area = (right1 - left1) * (bottom1 - top1)
    bb2_area = (right2 - left2) * (bottom2 - top2)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def is_unique_text_object(bbox_to_text, new_bbox, new_text):
    for bbox, text in bbox_to_text.items():
        if text == new_text:
            if get_iou(bbox, new_bbox) > 0.5:
                return False
    return True


def get_loc(img, bounding_box):
    img_height, img_width = img.shape[0], img.shape[1]
    left = int(bounding_box["Left"] * img_width)
    top = int(bounding_box["Top"] * img_height)
    right = int(left + (bounding_box["Width"] * img_width))
    bottom = int(top + (bounding_box["Height"] * img_height))
    return (left, top, right, bottom)


def check_regex(text, key):
    regexes = {
        "phone": "^(\\+\\d{1,2}\\s)?\\(?\\d{3}\\)?[\\s.-]?\\d{3}[\\s.-]?\\d{4}$",
        "price": "^\\$?\\d+[.,]\\d{1,2}$",
    }
    return bool(re.match(regexes[key], text))


def get_source_bytes(img: str):
    with open(img, "rb") as source_image:
        source_bytes = source_image.read()
    return source_bytes


def get_client():
    with open("../credentials.csv", "r") as _input:
        next(_input)
        reader = csv.reader(_input)
        for line in reader:
            access_key_id = line[2]
            secret_access_key = line[3]

    client = boto3.client(
        "rekognition",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="us-west-2",
    )
    return client


def get_environment(
    img_dirs: List[str], client, img_index, keys=[], prev_environment=None, max_faces=10
) -> Dict[str, Dict[str, Any]]:
    # The same face in a different photo has a different entry in the library,
    # but the SAME Index
    face_responses = []
    text_responses = []
    object_responses = []
    imgs = []
    for img_dir in img_dirs:
        if img_dir.endswith("DS_Store"):
            continue
        print("img_dir:", img_dir)
        face_response = client.index_faces(
            CollectionId="library2",
            Image={"Bytes": get_source_bytes(img_dir)},
            MaxFaces=max_faces,
            DetectionAttributes=["ALL"],
        )
        text_response = client.detect_text(Image={"Bytes": get_source_bytes(img_dir)})
        object_response = client.detect_labels(
            Image={"Bytes": get_source_bytes(img_dir)}, MaxLabels=100, MinConfidence=90
        )
        img = cv2.imread(img_dir, 1)
        face_responses.append(face_response)
        text_responses.append(text_response)
        object_responses.append(object_response)
        imgs.append(img)
    return get_details(
        face_responses,
        text_responses,
        object_responses,
        keys,
        imgs,
        client,
        img_index,
        prev_environment,
    )


def get_size(loc):
    left, top, right, bottom = loc
    return abs(left - right) * abs(top - bottom)


name_to_parent = {
    "bride": "person",
    "groom": "person",
}


def find_matching_obj(env, bbox, name):
    for obj in env.values():
        if obj["Type"] != "Object" or obj["Name"] != name:
            continue
        if get_iou(bbox, obj["Loc"]) > 0.95:
            return obj
    return None


def get_details(
    face_responses,
    text_responses,
    object_responses,
    keys: List[str],
    imgs,
    client,
    img_index,
    prev_environment=None,
    max_faces=10,
) -> Dict[str, Dict[str, Any]]:
    details_list = []
    details_maps = {}

    if not prev_environment:
        obj_count = 0
        img_count = 0
    else:
        obj_count = len(prev_environment)
        img_count = max(item["ImgIndex"] for item in prev_environment.values()) + 1
    for img_index, (face_response, text_response, img) in enumerate(
        zip(face_responses, text_responses, imgs)
    ):
        faces = face_response["FaceRecords"]
        text_objects = text_response["TextDetections"]
        objects = object_responses[0]["Labels"]
        for face in faces:
            details = face["FaceDetail"]
            face_hash = face["Face"]["FaceId"]
            details_map = {}
            details_map["Type"] = "Face"
            for key in keys:
                if key == "Emotions":
                    details_map[key] = []
                    emotion_list = details[key]
                    for emotion in emotion_list:
                        if emotion["Confidence"] > 90:
                            details_map[key].append(emotion["Type"])
                elif key == "AgeRange":
                    details_map[key] = details[key]
                else:
                    if details[key]["Value"] and details[key]["Confidence"] > 90:
                        # The value doesn't matter here
                        details_map[key] = True
            details_map["Loc"] = get_loc(img, face["Face"]["BoundingBox"])
            # Check if this face matches another face in the library
            if face_hash in face_hash_to_id:
                face_index = face_hash_to_id[face_hash]
            else:
                if prev_environment is None:
                    face_index = obj_count
                else:
                    search_response = client.search_faces(
                        CollectionId="library2",
                        FaceId=face_hash,
                        MaxFaces=max_faces,
                        FaceMatchThreshold=80,
                    )
                    hashes_to_indices = {
                        details["Hash"]: details["Index"]
                        for details in prev_environment.values()
                        if details["Type"] == "Face"
                    }
                    matched_face_hashes = [
                        item["Face"]["FaceId"]
                        for item in search_response["FaceMatches"]
                    ]
                    face_index = obj_count
                    for matched_face_hash in matched_face_hashes:
                        if matched_face_hash == face_hash:
                            continue
                        if matched_face_hash in hashes_to_indices:
                            face_index = hashes_to_indices[matched_face_hash]
                            break
            details_map["Index"] = face_index
            details_map["Hash"] = face_hash
            details_map["ImgIndex"] = img_index + img_count
            details_list.append(details_map)
            obj_count += 1
        bbox_to_text = {}
        for text_object in text_objects:
            if text_object["Confidence"] < 90:
                continue
            bbox = get_loc(img, text_object["Geometry"]["BoundingBox"])
            if not is_unique_text_object(
                bbox_to_text, bbox, text_object["DetectedText"]
            ):
                continue
            details_map = {}
            details_map["Type"] = "Text"
            text = text_object["DetectedText"]
            bbox_to_text[bbox] = text
            details_map["Text"] = text
            details_map["Loc"] = bbox
            if check_regex(text, "phone"):
                details_map["IsPhoneNumber"] = True
            if check_regex(text, "price"):
                details_map["IsPrice"] = True
            details_map["Index"] = obj_count
            details_map["ImgIndex"] = img_index + img_count
            details_list.append(details_map)
            obj_count += 1
        for obj in objects:
            for instance in obj["Instances"]:
                if instance["Confidence"] < 85:
                    continue
                # Remove redundant objects
                if obj["Name"] in {"Adult", "Child", "Man", "Male", "Woman", "Female"}:
                    continue
                details_map = {}
                details_map["Type"] = "Object"
                details_map["Name"] = obj["Name"]
                details_map["Loc"] = get_loc(img, instance["BoundingBox"])
                details_map["Index"] = obj_count
                details_map["ImgIndex"] = img_index + img_count
                details_list.append(details_map)
                obj_count += 1
        details_list.sort(key=lambda d: d["Loc"][0])
        details_maps = {}
        for i, details_map in enumerate(details_list):
            details_map["ObjPosInImgLeftToRight"] = i
            details_maps[i + len(prev_environment)] = details_map

    return details_maps
