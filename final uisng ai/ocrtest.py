import requests

files = {
    "file": open("test.jpeg", "rb")
}

r = requests.post(
    "http://localhost:5678/webhook-test/ocr",
    files=files
)


print(r.json())
