# TODO: Docs - python file that takes raw data and converts to csv.

from sklearn.model_selection import train_test_split
import csv
import json
import os

# gets correct directory path regardless of where you run script from
scripts_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(scripts_dir, "../data")

with open(os.path.join(data_dir, "x_train.txt"), "r", encoding="utf-8") as f:
    texts = f.readlines()
    texts = [text.strip() for text in texts]

with open(os.path.join(data_dir, "y_train.txt"), "r", encoding="utf-8") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]

with open(os.path.join(data_dir, "x_test.txt"), "r", encoding="utf-8") as f:
    test_texts = f.readlines()
    test_texts = [test_text.strip() for test_text in test_texts]

with open(os.path.join(data_dir, "y_test.txt"), encoding="utf-8") as f:
    test_labels = f.readlines()
    test_labels = [test_label.strip() for test_label in test_labels]

# hardcoded set of most recognizable languages
top_languages = {
    "eng", "fra", "spa", "deu", "zho",
    "ara", "rus", "jpn", "por", "ita",
    "nld", "pol", "tur", "swe", "kor",
    "fin", "dan", "nor", "hin", "vie",
    "ces", "hun", "ron", "ukr", "ind"
}

# filter by set
filtered_training_data = [(text, label) for (text, label) in zip(texts, labels) if label in top_languages] 
filtered_testing_data = [(test_text, test_label) for (test_text, test_label) in zip(test_texts, test_labels) if test_label in top_languages]

# create two way mappings - guaranteed, stable order with sorted
label_to_int_mapping = {lang: i for i, lang in enumerate(sorted(top_languages))}
int_to_label_mapping = {i: lang for i, lang in enumerate(sorted(top_languages))}

# apply mappings to filtered training data
filtered_training_data = [(text, label_to_int_mapping[label]) for (text, label) in filtered_training_data]
filtered_testing_data = [(test_text, label_to_int_mapping[test_label]) for (test_text, test_label) in filtered_testing_data]

# create train-val split
filtered_training_data, filtered_validation_data = train_test_split(filtered_training_data, test_size=0.1, random_state=42)

# create csv files
with open(os.path.join(data_dir, "training.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(("text", "label"))
    writer.writerows(filtered_training_data)

with open(os.path.join(data_dir, "validation.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(("text", "label"))
    writer.writerows(filtered_validation_data)

with open(os.path.join(data_dir, "testing.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(("text", "label"))
    writer.writerows(filtered_testing_data)

# dump to json
with open(os.path.join(data_dir, "mappings.json"), "w", encoding="utf-8") as f:
    json.dump({
        "label_to_int": label_to_int_mapping,
        "int_to_label": int_to_label_mapping,
    }, f)

# remove raw .txt files
# os.remove(os.path.join(data_dir, "x_test.txt"))
# os.remove(os.path.join(data_dir, "x_train.txt"))
# os.remove(os.path.join(data_dir, "y_test.txt"))
# os.remove(os.path.join(data_dir, "y_train.txt"))