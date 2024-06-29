## About

This is a project that shows how to fine tune OpenAI's `gpt-3.5-turbo` model on a custom dataset. The dataset used in this project is a collection of a few news articles. The model is fine-tuned on this dataset to classify the news articles as important or not important. For this project, important news articles are those that are related to the COVID-19 pandemic.

## Dataset

The example dataset only contains a few news articles. The CSV file is located [here](./data/news.csv). Following are the columns in the dataset:

- `id`: The unique identifier of the news article.
- `title`: The title of the news article.
- `subtitle`: The subtitle of the news article.
- `text`: The text of the news article.
- `journal`: The journal that published the news article.
- `important`: The label of the news article. It is `1` if the news article is important and `0` otherwise.

One example from the dataset will be used as a validation dataset, and the rest will be used for training. In a production environment, you should of course use a larger dataset. Check OpenAI's [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) for more information.

## Codebase

Here is a documentation of the codebase.

### Transform the dataset

First, the dataset is transformed into a format that can be used for fine-tuning the model. The [transform_dataset.py](./src/transform_dataset.py) script is used for this purpose. The script reads the CSV file and creates a JSON file that contains the training and validation datasets.

Here is how you can run the script:

```bash
python src/transform_dataset.py
```

### Fine-tune the model

The script [fine_tune.py](./src/fine_tune.py) fine-tunes the model on the transformed dataset. The script uses the `gpt-3.5-turbo` model from OpenAI's API.

Here is how you can run the script:

```bash
python src/fine_tune.py
```

The script outputs should look like this:

```text
Train file: <file_id>
Validation file: <file_id>
Job status: validating_files
Job status: running
Job status: running
Fine tuning completed. Model: <fine_tuned_model_id>
```

In OpenAI's dashboard, the fine-tuned model you can see more details about the fine-tuned model, as shown below:

[![Fine-tuned model](./doc/fine-tuned-model.png)](./doc/fine-tuned-model.png)
