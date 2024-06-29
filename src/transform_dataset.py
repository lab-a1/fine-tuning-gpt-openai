import os
import csv
import json
from typing import Dict, Generator, List, Tuple

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
TRAIN_INDEX = 1
VALIDATION_INDEX = 2
PROMPT = "You are a system that understands news that are important. If the news is important, classify it as 'important'. Otherwise, classify it as 'not important'. To classify the news, you should consider the core of the news, or what it really means. You should not use other classifications. Otherwise, the answer will be considered invalid."
INPUT_PROMPT = """Title: {title}

Subtitle: {subtitle}

Body: {body}"""


def read_csv_file(file_path: str) -> Generator[Tuple[int, Dict[str, str]], None, None]:
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            yield i, row


def important_label(value: str) -> str:
    return "important" if value == "1" else "not important"


def convert_text_to_openai_format(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    openai_data = []
    for row in data:
        openai_data.append(
            {
                "messages": [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": INPUT_PROMPT.format(**row)},
                    {"role": "assistant", "content": important_label(row["important"])},
                ]
            }
        )
    return openai_data


def store_openai_data(file_path: str, data: List[Dict[str, str]]) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


# load csv file
csv_path = os.path.join(FILE_PATH, "../data/news.csv")

train_data = []
validation_data = []
for i, row in read_csv_file(csv_path):
    if i <= TRAIN_INDEX:
        train_data.append(row)
    elif i <= VALIDATION_INDEX:
        validation_data.append(row)

# convert data to openai format
train_openai_data = convert_text_to_openai_format(train_data)
validation_openai_data = convert_text_to_openai_format(validation_data)

# store data in openai format
train_json_path = os.path.join(FILE_PATH, "../data/train.json")
validation_json_path = os.path.join(FILE_PATH, "../data/validation.json")

store_openai_data(train_json_path, train_openai_data)
store_openai_data(validation_json_path, validation_openai_data)
