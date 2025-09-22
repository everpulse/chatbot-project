# v1-simple/chatbot.py
# Simple chatbot using CountVectorizer + MultinomialNB
import json
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

DATA_FILE = Path(__file__).with_name("dataset.json")

if DATA_FILE.exists():
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
    questions = list(data.keys())
    answers = list(data.values())
else:
    questions = ["سلام", "حالت چطوره", "اسم تو چیه", "خداحافظ", "چیکار میکنی"]
    answers = ["سلام! خوش اومدی.", "خوبم، ممنون! تو چطوری؟", "من یه چت‌بات ساده‌ام.", "فعلاً خداحافظ، موفق باشی!", "دارم با تو حرف می‌زنم 😎"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

model = MultinomialNB()
model.fit(X, list(range(len(answers))))

print("چت‌بات آماده‌ست! (برای خروج 'exit' بزن)")

while True:
    user_input = input("تو: ").strip()
    if user_input.lower() == "exit":
        print("چت‌بات: خداحافظ! 😊")
        break
    user_vec = vectorizer.transform([user_input])
    pred = model.predict(user_vec)[0]
    print("چت‌بات:", answers[pred])
