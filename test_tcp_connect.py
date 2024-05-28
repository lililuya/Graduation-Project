import os
import time
import socket

"""服务端需要将生成的framelist以及音频传到客户端"""

if __name__=="__main__":
    from whole_pipeline_socket import metahuman
    # 这个地方需要对模型进行初始化
    infer_model = metahuman()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '175.24.178.88'  # 服务器主机地址
    port = 8984 # 服务器端口, 本地的端口
    server_socket.bind((host, port))
    # 监听传入的连接，允许最大的连接数为1
    server_socket.listen(1)
    # 接收请求
    client_socket, addr = server_socket.accept()
    
    while True:  # 每次接收四个字节
        size_data = client_socket.recv(4)  # 接收4字节的大小信息
        audio_size = struct.unpack('!I', size_data)[0]  # 解码为无符号整数
        print("接收音频数据大小：%d bytes" % audio_size)

        # 接收音频数据
        audio_data = b""
        while len(audio_data) < audio_size:
            data = client_socket.recv(audio_size - len(audio_data))
            if not data:
                break
            audio_data += data

        # 保存接收到的音频文件
        output_file = 'received_audio.wav'
        with open(output_file, 'wb') as file:
            file.write(audio_data)

        print("音频文件已保存：%s" % output_file)

        # 这个地方不止有音频的输入
        audio_path, frame_list = infer_model(output_file)
        frames = []
        
        with open(audio_path, 'rb') as file:
            file_data = file.read()
            file_size = len(file_data)
            client_socket.sendall(file_size.to_bytes(4, 'big'))
        client_socket.sendall(file_data)
        print("反馈音频文件已发送：%s" % audio_path)
        client_socket.sendall(len(frame_list).to_bytes(4, 'big'))  # 将音频的长度信息传递出去
        
        # 对里面的每帧数据进行发送
        for frame in frames:
            data_bytes = frame.tobytes()
            file_size = len(data_bytes)
            client_socket.sendall(file_size.to_bytes(4, 'big'))
            client_socket.sendall(data_bytes)

    # 关闭与客户端的连接
    client_socket.close()
    print("与客户端的连接已关闭")

    # 关闭服务器socket
    server_socket.close()
    print("服务器已关闭")