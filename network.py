import socket
import sys
import json


class gameNetwork:

    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = "localhost"
        self.port = 5555
        self.addr = (self.host, self.port)
        self.id = self.connect()

    def connect(self):
        self.client.connect(self.addr)
        return self.client.recv(2048).decode()

    def send(self, data):
        """
        :param data: json object
        :return: str
        """
        try:
            p_data = json.dumps(data)
            self.client.sendall(bytes(p_data, encoding='utf-8'))
            reply = self.client.recv(2048).decode()

            print("Game id: {}".format(data['id']))
            with open('data_{}_{}'.format(data['id'], data['timestep']) + '.json', 'w') as outfile:
                json.dump(data, outfile)

            return reply
        except socket.error as e:
            return str(e)