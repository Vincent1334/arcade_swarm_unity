import time
import json
import socket
import requests
import threading
import random

HEADER = 128
HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65431        # Port to listen on (non-privileged ports are > 1023)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))

def send_random(id, width, height, drones):
    api_server = "http://localhost:8000"
    headers={'Content-type':'application/json', 'Accept':'application/json'}

    for e in range(0, 50):
        belief = [[ random.randint(0, height - 1), random.randint(0, width - 1)] for d in range(0, drones) ]
        confidence = [[random.random() for j in range(0, width)] for i in range(0, height)]

        data = {
            "config": {
                "belief": belief,
                "confidence": confidence
            }
        }

        r = requests.post(api_server + "/api/v1/simulations/" + id + "/timestep/" + str(e), data=json.dumps(data),headers=headers)
        time.sleep(5)


def handle_client(conn, addr):
    connected = True

    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)

        if msg_length:
            msg_length = int(msg_length)

            msg = conn.recv(msg_length).decode(FORMAT)

            if msg == DISCONNECT_MESSAGE:
                connected = False

            data = json.loads(msg)

            print(data)
            if data["action"] == "setup":
                send_random(data["sim_id"], data["width"], data["height"], data["drones"])

    conn.close()

def start():
    server.listen()
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()

print("Starting server ...")
start()
