# Pet Classifier CNN: Cat vs Dog

A convolutional neural network built **from scratch in PyTorch** that tells cats from dogs, trained on the Oxford-IIIT Pet dataset. The project covers the full pipeline: data exploration, four training experiments tracked in Weights & Biases, evaluation with error analysis on a held-out test set, a model-size optimization, and a FastAPI + OpenCV inference API.

**Test result (final model):** accuracy **0.797**, macro F1 **0.761**

---

## Project structure

```
pet-classifier-cnn/
├── api/                     FastAPI + OpenCV inference API
│   ├── api.py               endpoints: /predict and /predict/image
│   ├── predict.py           loads the model, preprocesses the image, predicts
│   └── batchnorm_dropout_cnn.py   model architecture
├── models/
│   └── batchnorm_dropout_cnn.pt   trained final model (18.4 MB)
├── notebooks/
│   ├── data_exploratory_analysis.ipynb   EDA (also downloads the dataset)
│   ├── baseline_cnn.ipynb                run 1
│   ├── batchnorm_dropout_cnn.ipynb       run 2 (final model)
│   ├── augmented_cnn.ipynb               run 3
│   ├── compact_cnn.ipynb                 run 4 (optimization)
│   └── model_inference.ipynb             test-set evaluation + error analysis
├── pyproject.toml / uv.lock  dependencies (managed with uv)
└── README.md
```

## Setup and how to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone https://github.com/Nestor-Dzadzamia/pet-classifier-cnn.git
cd pet-classifier-cnn
uv sync
```

**Dataset:** run the first cell of `notebooks/data_exploratory_analysis.ipynb`. It downloads Oxford-IIIT Pet (~800 MB) into `data/` with `torchvision.datasets.OxfordIIITPet`.

**Training notebooks** were run on a Kaggle GPU (T4). Each one downloads the dataset itself and logs to W&B (`wandb.login()` asks for an API key).

**Inference notebook** downloads the final model from my W&B project. Without access to it, load the local file instead:
```python
model.load_state_dict(torch.load("../models/batchnorm_dropout_cnn.pt", map_location="cpu"))
```

### Run the API

```bash
uv run uvicorn api.api:app --reload
```

Open **http://127.0.0.1:8000/docs**, choose an endpoint, click *Try it out*, upload a photo and click *Execute*.

| Endpoint | Returns |
|---|---|
| `POST /predict` | JSON, e.g. `{"label": "dog", "confidence": 0.686}` |
| `POST /predict/image` | the uploaded photo with the prediction drawn on it by OpenCV (PNG) |

From the terminal:
```bash
curl -F "file=@my_dog.jpg" http://127.0.0.1:8000/predict
```

Uploading a file that is not an image returns `400: provided file is not an image`.

---

## 1. Problem

Binary image classification: given a photo of a pet, predict **cat** or **dog**. I built the CNN from scratch, without pretrained models, so that every part of the model (layers, shapes, regularization) is my own and I can explain it.

## 2. Data

**Oxford-IIIT Pet** (one of the datasets listed in the task): 7,349 photos of 37 breeds (12 cat breeds, 25 dog breeds), relabeled to 2 classes. Breed labels were kept for error analysis.

| Split | Photos | Use |
|---|---|---|
| train | 2,944 | training (80% of the official trainval list) |
| validation | 736 | picking the best epoch and comparing runs (20% of trainval) |
| test | 3,669 | official test list, used **once** at the end on the final model |

The train/validation split is **stratified by breed**, so every breed is split 80/20 and the cat/dog ratio is the same in both sets.

**EDA findings** (`data_exploratory_analysis.ipynb`, on trainval only):
- **Imbalanced classes:** 1,188 cats (32%) vs 2,492 dogs (68%). Always answering "dog" already scores 67.7% accuracy, so accuracy alone is misleading. I report **macro F1** and a confusion matrix.
- **Breeds are balanced** (93–100 photos each). The class imbalance comes from having 12 cat breeds vs 25 dog breeds.
- **Image sizes vary a lot** (from about 110 px to 3,264 px), so every image is resized to **128×128**.
- **2 images are RGBA** (4 channels), so every image is converted to RGB when loading.

Preprocessing: `Resize((128, 128))` → `ToTensor()` (pixels scaled to 0–1).

## 3. Model and experiments

All runs: CrossEntropyLoss, Adam (lr 1e-3), batch size 64, 20 epochs. After each epoch the model is saved only if validation macro F1 improved, so the **best epoch** is kept. All runs are tracked in W&B: [pet-classifier-cnn](https://wandb.ai/nestor_dzadzamia-free-university-of-tbilisi-/pet-classifier-cnn).

| Run | What changed | Parameters |
|---|---|---|
| 1. Baseline | 2 conv layers (16→32) + Linear(32768→128) + Linear(128→2) | 4,199,778 |
| 2. BatchNorm + Dropout | 4 conv blocks (32→64→128→256) with BatchNorm, Linear(16384→256), Dropout 0.5 | 4,584,450 |
| 3. + Augmentation | same model as run 2, training images randomly cropped (0.7–1.0), flipped, rotated (±15°) and brightened (±20%) | 4,584,450 |
| 4. Compact | half the channels (16→32→64→128) + Global Average Pooling + Linear(128→2) | 98,178 |

### Results (validation, best epoch)

| Run | Best epoch | Train acc | Val acc | **Val F1** |
|---|---|---|---|---|
| 1. Baseline | 8 | 0.844 | 0.747 | 0.706 |
| **2. BatchNorm + Dropout** | 18 | 0.829 | **0.808** | **0.778** |
| 3. + Augmentation | 11 | 0.735 | 0.766 | 0.740 |
| 4. Compact | 19 | 0.762 | 0.736 | 0.720 |

**What each run showed:**
- **Run 1 overfits.** Train accuracy reaches 100% by epoch 20 while validation stays around 75%, and validation loss climbs from 0.52 to 1.59. The model memorizes the training photos. Most of its parameters (4.19M of 4.2M) sit in one big Linear layer, which makes memorizing easy.
- **Run 2 fixes the overfitting.** With BatchNorm and Dropout, train and validation accuracy stay close (0.83 vs 0.81) and F1 improves by 7 points. This is the **final model**.
- **Run 3: augmentation made it worse.** Augmentation fights overfitting, but run 2 was barely overfitting anymore. The harder training data meant the model was still learning after 20 epochs (train accuracy 0.79 and rising). It is underfitting within this training budget.
- **Run 4: 47× fewer parameters for 2 points of F1**

## 4. Evaluation on the test set

The final model (run 2) on the 3,669 test photos:

| | Accuracy | Macro F1 |
|---|---|---|
| Always predict "dog" | 0.678 | 0.404 |
| **BatchNorm + Dropout CNN** | **0.797** | **0.761** |

| Class | Precision | Recall | F1 | Photos |
|---|---|---|---|---|
| cat | 0.71 | 0.64 | 0.67 | 1,183 |
| dog | 0.83 | 0.87 | 0.85 | 2,486 |

**Confusion matrix:**

| | predicted cat | predicted dog |
|---|---|---|
| **true cat** | 752 | 431 |
| **true dog** | 313 | 2,173 |

Test F1 (0.761) is slightly below validation F1 (0.778). That is expected, because the best epoch was chosen on validation, which makes the validation score a little optimistic. The gap is small, so the model generalizes to unseen photos.

## 5. Error analysis

**Cats are the weak class.** Cat recall is 0.64: about 1 in 3 cats is predicted as a dog (431 of 1,183). Dogs make up 68% of the training data, so when the model is unsure it leans towards "dog".

**All 10 worst breeds are cats:**

| Breed | Test accuracy |
|---|---|
| Bombay | 0.56 |
| Bengal | 0.57 |
| Birman | 0.57 |
| Abyssinian | 0.59 |
| Russian Blue | 0.61 |
| Sphynx | 0.62 |
| British Shorthair | 0.63 |
| Ragdoll | 0.63 |
| Siamese | 0.64 |
| Persian | 0.68 |

**Patterns in the misclassified photos:**
- **Cat-like dogs predicted as cats:** keeshond, scottish terrier, japanese chin (fluffy fur, pointy ears or flat faces).
- **Dark or uniform fur:** Bombay (black) and Russian Blue (grey) give the model very little fur texture to work with.
- **The pet is small in the frame:** after resizing to 128×128, a cat on a flower pot or a bed is only a few pixels and the background dominates.
- **Unusual looks or pose:** a hairless Sphynx, a shiba curled up asleep with its face hidden, a dog held by a person.

## 6. Optimization (before / after)

Run 4 makes the model smaller by replacing the big classifier. **Global Average Pooling** averages each of the 128 final feature maps into one number, so only 128 values go into the Linear layer instead of 16,384, and the channel count is halved in every conv layer.

| | BatchNormDropoutCNN (before) | CompactCNN (after) |
|---|---|---|
| Parameters | 4,584,450 | **98,178** (47× fewer) |
| Model file size | 18.4 MB | **0.4 MB** (46× smaller) |
| Val F1, same training setup (with augmentation) | 0.740 | 0.720 |

Size and parameter count depend only on the architecture. The fair accuracy comparison is run 3 vs run 4, because both used the same training setup (augmentation) and only the architecture changed. The result is **47× fewer parameters for −2 points of F1**. I kept the bigger model for the API because accuracy matters more than size for this demo, but the compact model would suit a phone or other small device.

## 7. Problems faced

- **Overfitting** in the baseline (100% train vs 75% validation), solved with BatchNorm and Dropout.
- **Augmentation hurt** within 20 epochs. It needs longer training to pay off.
- **Noisy results:** I trained the run 2 setup three times and got validation F1 of 0.797, 0.802 and 0.778. Differences under about 2–3 points between experiments can be random. Validation scores also jump between epochs (only 736 photos), which is why the best epoch is saved instead of the last one.
- **Class imbalance** shows up directly in the results as weak cat recall.

## 8. Strengths and weaknesses

**Strengths**
- Every part is built and explained by me: the architecture, the shapes, the training loop and the metrics.
- A clean experiment story with honest results, including a negative one (augmentation).
- The test set was used only once, on the final model.
- A working API with input validation and an image output.

**Weaknesses**
- About 80% test accuracy, well below what a pretrained model would reach on this task.
- Weak on cats (recall 0.64), especially dark, small-in-frame or unusual-looking cats.
- The model is often not confident even when correct (confidence around 50–70%).

## 9. Future improvements

- **Transfer learning:** fine-tune a pretrained model such as ResNet18. It already knows fur, eyes and faces from ImageNet, so it would very likely beat this from-scratch CNN by a wide margin.
- **Class weights** in the loss (`nn.CrossEntropyLoss(weight=...)`), so mistakes on cats cost more.
- **Longer training with a learning-rate schedule**, which would also let augmentation help.
- **Repeat each experiment several times** and average the results, to separate real improvements from noise.
