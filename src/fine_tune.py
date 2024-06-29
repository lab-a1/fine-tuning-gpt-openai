import os
import sys
import time
from typing import Optional
from openai import OpenAI

FILE_PATH = os.path.abspath(os.path.dirname(__file__))


def file_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def main(train_file_id: Optional[str] = None, validation_file_id: Optional[str] = None):
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI()

    # create file to fine tune
    if train_file_id is None:
        train_json_path = os.path.join(FILE_PATH, "../data/train.json")
        train_file = client.files.create(
            file=open(train_json_path, "rb"), purpose="fine-tune"
        )
    else:
        train_file = client.files.retrieve(train_file_id)

    if validation_file_id is None:
        validation_json_path = os.path.join(FILE_PATH, "../data/validation.json")
        validation_file = client.files.create(
            file=open(validation_json_path, "rb"), purpose="fine-tune"
        )
    else:
        validation_file = client.files.retrieve(validation_file_id)

    print(f"Train file: {train_file.id}")
    print(f"Validation file: {validation_file.id}")

    train_fine_tuning = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=validation_file.id,
        model="gpt-3.5-turbo",
        hyperparameters={
            "n_epochs": 2,
            "batch_size": 10,
        },
    )

    # wait for fine tuning to complete
    while True:
        job = client.fine_tuning.jobs.retrieve(train_fine_tuning.id)
        if job.status == "succeeded":
            break
        elif job.status == "failed":
            print("Job failed")
            break
        elif job.status == "cancelled":
            print("Job cancelled")
            break
        else:
            print(f"Job status: {job.status}")
            time.sleep(60)

    print(f"Fine tuning completed. Model: {job.fine_tuned_model}")


if __name__ == "__main__":
    # once data is uploaded to openai, you can choose to pass the files ids as arguments
    # to avoid uploading the files again
    train_file_id = sys.argv[1] if len(sys.argv) > 1 else None
    validation_file_id = sys.argv[2] if len(sys.argv) > 2 else None

    main(train_file_id, validation_file_id)
