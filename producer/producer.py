import random
import json
import time
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:29092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

for _ in range(500):

    user_id = random.randint(1, 30)

    base = random.uniform(20, 200)

    spike = random.random()

    # comportamento normal
    amount = base

    # comportamento fraudulento (spike raro)
    if spike > 0.92:
        amount = random.uniform(350, 600)

    # regra realista de fraude
    fraud = 1 if amount > 400 else 0

    transaction = {
        "user_id": user_id,
        "amount": round(amount, 2),
        "fraud": fraud
    }

    producer.send("transactions", transaction)

    print("Sent:", transaction)
    time.sleep(0.3)
