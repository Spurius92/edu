import socket

HEADERSIZE = 10
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1234))

while True:
    full_msg = ''
    new_msg = True
    while True:
        msg = s.recv(14)
        if new_msg:
            # print("new msg len:", len(msg[:HEADERSIZE]))
            msglen = int(len(msg[:HEADERSIZE]))
            new_msg = False

        print(f"full message length: {msglen}")

        full_msg += msg.decode("utf-8")

        print(len(full_msg))

        if len(full_msg) - HEADERSIZE == msglen:
            print('Full message recieved!')
            print(full_msg[HEADERSIZE:])
            new_msg = True
            full_msg = ''

print(full_msg)
