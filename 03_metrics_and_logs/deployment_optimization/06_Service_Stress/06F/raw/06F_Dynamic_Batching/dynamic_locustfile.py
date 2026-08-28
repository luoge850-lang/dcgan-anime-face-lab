import itertools
import time
from locust import HttpUser, task

SEEDS = itertools.count(int(time.time()) % 1000000)

class GeneratorUser(HttpUser):
    wait_time = lambda self: 0.0

    @task
    def generate(self):
        seed = next(SEEDS)
        with self.client.post("/generate", json={"seed": seed}, name="POST /generate", timeout=30, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
            elif not response.headers.get("content-type", "").startswith("image/png"):
                response.failure("response is not image/png")
            elif not response.content:
                response.failure("empty response")
            else:
                response.success()
