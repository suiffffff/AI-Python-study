from pymongo import MongoClient
import pandas as pd


client = MongoClient("mongodb://localhost:27017/")
db = client["pet_database"]
collection = db["dog_stats"]

dog_data = [
    {"名字": "旺财", "使用次数": 854, "品种": "中华田园犬", "特征": ["忠诚", "看家"]},
    {"名字": "大黄", "使用次数": 632, "品种": "金毛", "特征": ["温顺", "贪吃"]},
    {"名字": "小黑", "使用次数": 415, "品种": "拉布拉多", "特征": ["聪明", "导盲"]},
    {"名字": "豆豆", "使用次数": 920, "品种": "贵宾犬", "特征": ["活泼", "不掉毛"]},
    {"名字": "胖虎", "使用次数": 128, "品种": "法斗", "特征": ["打呼噜", "憨厚"]}
]


collection.delete_many({})


result = collection.insert_many(dog_data)
print(f"✅ 成功插入了 {len(result.inserted_ids)} 条狗狗数据！")

query = {"使用次数": {"$gt": 500}}

for dog in collection.find(query):
    print(f"- {dog['名字']} ({dog['品种']})：被使用了 {dog['使用次数']} 次")

data=list(collection.find())
t1=pd.DataFrame(data)
print(t1)
client.close()