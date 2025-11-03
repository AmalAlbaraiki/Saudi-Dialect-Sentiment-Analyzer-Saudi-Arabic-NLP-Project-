# Saudi-Dialect-Sentiment-Analyzer-Saudi-Arabic-NLP-Project-


## 🧠 مشروع: محلل المشاعر باللهجة السعودية البيضاء

### *(Saudi Dialect Sentiment Analyzer — Saudi Arabic NLP Project)*

---

### 🩵 مقدمة :

الذكاء الاصطناعي (AI) هو فرع من علوم الحاسب يهدف إلى جعل الأجهزة والبرامج “تفكر” وتتعلم مثل الإنسان.
يعني بدل ما نعطي الجهاز أوامر محددة، نخليه **يتعلم الأنماط** من البيانات ويستنتج النتيجة بنفسه.

من مجالات الذكاء الاصطناعي المعروفة:

* **الرؤية الحاسوبية (Computer Vision)** → لفهم الصور والفيديوهات.
* **معالجة اللغة الطبيعية (NLP)** → لفهم النصوص والكلام البشري.
* **التعلم الآلي (Machine Learning)** → لتدريب النماذج على التنبؤ أو التصنيف.

مشروعنا هنا يندرج تحت مجال **معالجة اللغة الطبيعية (NLP)**.
الفكرة أننا نعلّم الكمبيوتر **يفهم لهجتنا السعودية البيضاء** ويستنتج مشاعر النصوص — هل الكلام إيجابي، سلبي، أو محايد.

---

### 🎯 الهدف من المشروع:

إنشاء نموذج ذكاء اصطناعي يستطيع تحليل النصوص السعودية باللهجة اليومية،
ويتعرف على *نغمة الكلام* — هل هي إيجابية، سلبية، أو محايدة — تمامًا مثل طريقة فهم البشر للمحادثات.

---

### 🧩 الأدوات المستخدمة:

* **Python**
* **TensorFlow / Keras** لتصميم النموذج العصبي
* **pandas / numpy** لمعالجة البيانات
* **scikit-learn** لترميز الفئات وتقسيم البيانات

---

### ⚙️ خطوات العمل:

1. **جمع البيانات:**
   إنشاء مجموعة بيانات تحتوي على جمل سعودية بيضاء مثل:
   “الخدمة ممتازة والله” → إيجابي
   “التطبيق بطئ مرة” → سلبي

2. **معالجة النصوص:**
   تحويل الجمل إلى أرقام باستخدام Tokenization وPadding.

3. **بناء النموذج:**
   تصميم شبكة عصبية بسيطة من نوع LSTM لتتعلم العلاقات بين الكلمات والمشاعر.

4. **تدريب النموذج:**
   تدريب الذكاء الاصطناعي على البيانات حتى يتقن التصنيف بدقة عالية.

5. **الاختبار:**
   تجربة النموذج على جمل جديدة لم يرها من قبل لقياس دقته.

---

### 💬 أمثلة على التوقعات:

| النص                         | النتيجة   |
| ---------------------------- | --------- |
| “الخدمة ممتازة والله”        | إيجابي 👍 |
| “مره تأخر التوصيل وما رديتو” | سلبي 👎   |
| “عادي الخدمة متوسطة”         | محايد 😐  |

---

### 🔍 مستقبل المشروع:

النسخة الحالية تتعامل مع **اللهجة السعودية البيضاء**.
النسخ القادمة ستركز على اللهجات المحلية (نجدية، حجازية، جنوبية...)
بحيث يتعلم الذكاء الاصطناعي الفروق في التعبير بين كل منطقة في السعودية 🇸🇦

---

### 📚 عن المطوّرة:

المشروع من إعداد **أمل  البريكي**،
 **علوم حاسب —  مهندسة برمجيات**،
مهتمة بتطبيقات الذكاء الاصطناعي والروبوتات التفاعلية.
تعمل على مشاريع تجمع بين *التعلم الآلي، الرؤية الحاسوبية، ومعالجة اللغة الطبيعية.*

---
---

## 🇬🇧 **Saudi Dialect Sentiment Analyzer — (Saudi Arabic NLP Project)**

---

### 🧠 **Introduction**

Artificial Intelligence (AI) is a field of computer science that aims to enable machines to think and learn like humans.
Instead of giving the machine exact instructions, we train it with data so it can identify **patterns** and make predictions on its own.

This project focuses on **Natural Language Processing (NLP)** — a branch of AI that helps machines understand human language.
Here, we train a model to **analyze Saudi Arabic dialect** (specifically the “white” dialect used commonly across Saudi Arabia)
and classify text sentiment as **positive**, **negative**, or **neutral**.

---

### 🎯 **Project Goal**

To build an AI model that understands Saudi Arabic text and predicts whether the sentiment expressed in the text is positive, negative, or neutral —
similar to how humans sense tone and emotion in daily speech.

---

### 🧩 **Technologies Used**

* **Python 3**
* **TensorFlow / Keras** — for building and training the deep learning model
* **pandas / numpy** — for data preprocessing
* **scikit-learn** — for encoding and splitting data
* **Google Colab** — for training and experimentation

---

### ⚙️ **How It Works**

1. **Data Preparation:**
   A small sample dataset is created using real Saudi Arabic expressions:

   ```text
   "الخدمة ممتازة والله" → Positive  
   "التطبيق بطئ مرة" → Negative  
   "عادي مو ذاك الزود" → Neutral  
   ```

2. **Text Preprocessing:**
   Convert words into numeric sequences using `Tokenizer` and `pad_sequences`.

3. **Model Building:**
   Create an LSTM neural network that learns relationships between words and emotions.

4. **Training & Evaluation:**
   Train the model using labeled examples, then test it on unseen text.

---

### 💻 **Example Code**

Here’s a minimal version of the working model (can be run directly in **Google Colab**):

```python
# Saudi Dialect Sentiment Analyzer
# Developer: Amal Al-Buraiki

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# --- 1. Training Data (Saudi Dialect Sentences) ---
data = {
    "text": [
        "الخدمة ممتازة والله",
        "التطبيق بطئ مرة",
        "السعر مناسب جدًا",
        "ما انصح احد بالتعامل معهم",
        "عادي مو ذاك الزود",
        "التوصيل سريع ومريح",
        "واجهت مشاكل في الدفع",
        "الدعم الفني متعاون",
        "تجربة سيئة بصراحة",
        "مره اعجبني التعامل"
    ],
    "sentiment": [
        "positive",
        "negative",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive"
    ]
}

df = pd.DataFrame(data)

# --- 2. Preprocess Data ---
X = df["text"].values
y = df["sentiment"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, maxlen=10, padding="post")

# --- 3. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(padded, y_categorical, test_size=0.2, random_state=42)

# --- 4. Build Model ---
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    LSTM(64, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# --- 5. Train Model ---
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose=1)

# --- 6. Prediction Function ---
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=10, padding="post")
    result = model.predict(padded_seq)
    label = label_encoder.inverse_transform([np.argmax(result)])[0]
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {label}")

# --- 7. Try it! ---
predict_sentiment("الخدمة كانت سريعة وممتازة")
predict_sentiment("مره تأخر التوصيل وما رديتو")
predict_sentiment("عادي الخدمة متوسطة")
```

---

### 💬 **Example Output**

```
Text: الخدمة كانت سريعة وممتازة  
Predicted Sentiment: positive  

Text: مره تأخر التوصيل وما رديتو  
Predicted Sentiment: negative  

Text: عادي الخدمة متوسطة  
Predicted Sentiment: neutral  
```

---

### 🚀 **Future Work**

* Expand dataset to include **real Saudi tweets and reviews**.
* Add **dialect-specific** models (Najdi, Hijazi, Southern, etc.).
* Create a **web interface** for public testing.

---

### 👩‍💻 **About the Developer**

**Amal  Al-Baraiki**
Computer Science  — Software Engineering .
Passionate about Artificial Intelligence, NLP, and Robotics.
Experienced in AI projects including **interactive robots** and **machine learning applications**.

---


