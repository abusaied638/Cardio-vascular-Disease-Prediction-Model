import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ১. তৈরি করা CSV ফাইলটি লোড করা
df = pd.read_csv('cardio_data.csv')

# ২. ইনপুট ফিচার (X) এবং আউটপুট টার্গেট (y) আলাদা করা
X = df.drop(columns=['id', 'cardio']) # id এবং target বাদ দিয়ে বাকি ১১টি ফিচার [১.৩.২]
y = df['cardio'] # হৃদরোগের উপস্থিতি (0 বা 1) [১.৩.২]

# ৩. ডেটাকে ট্রেনিং এবং টেস্টিং সেটে ভাগ করা (৮০% ট্রেনিং, ২০% টেস্টিং)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ৪. গবেষণাপত্রের মতো ১০০টি গাছ (n_estimators=100) নিয়ে র‍্যান্ডম ফরেস্ট মডেল তৈরি [০.১.৪]
model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=1) # [০.১.৪]
model.fit(X_train, y_train)

# ৫. টেস্ট ডেটার ওপর প্রেডিকশন এবং ফলাফল মূল্যায়ন
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"মডেলের সঠিকতা (Accuracy): {accuracy * 100:.2f}%\n")
print("বিস্তারিত রিপোর্ট:")
print(classification_report(y_test, y_pred))
