import pickle
import os

for model_name in ["isl_model.pkl", "isl_model_one_hand.pkl"]:
    if os.path.exists(model_name):
        print(f"=== {model_name} ===")
        try:
            with open(model_name, "rb") as f:
                model = pickle.load(f)
            print(f"Type: {type(model)}")
            if hasattr(model, "n_features_in_"):
                print(f"Features: {model.n_features_in_}")
            if hasattr(model, "classes_"):
                print(f"Classes ({len(model.classes_)}): {list(model.classes_)[:10]}...")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    else:
        print(f"{model_name} does not exist.")

for label_name in ["labels.pkl", "labels_one_hand.pkl"]:
    if os.path.exists(label_name):
        print(f"=== {label_name} ===")
        try:
            with open(label_name, "rb") as f:
                labels = pickle.load(f)
            print(f"Labels ({len(labels)}): {labels[:10]}...")
        except Exception as e:
            print(f"Error loading {label_name}: {e}")
