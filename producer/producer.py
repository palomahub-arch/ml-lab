from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

while True:
    event = {
        "user_id": random.randint(1, 100),
        "amount": round(random.uniform(10, 500), 2),
        "fraud": random.choice([0, 1])
    }

    producer.send("transactions", event)
    print("Sent:", event)

    time.sleep(1)
