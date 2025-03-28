import json
import os
import socket
import threading
import time

import requests
from sshtunnel import SSHTunnelForwarder


def check_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False  # 端口未被占用
        except socket.error:
            return True  # 端口已被占用


def setup_vllm_servers(vllm_urls, local_port, remote_port):
    if len(vllm_urls) != len(local_port) or len(vllm_urls) != len(remote_port):
        raise ValueError("Number of vLLM URLs must match number of local ports and remote ports")
    """
    Setup vLLM servers by checking URLs, ports and establishing SSH tunnels.
    Returns list of active SSH tunnels.
    """
    # Check number of vLLM URLs

    if len(vllm_urls) > 2:
        print("Warning: More than 2 vLLM URLs provided. Only first 2 will be used.")
        vllm_urls = vllm_urls[:2]
    elif len(vllm_urls) < 1:
        raise ValueError("At least one vLLM URL must be provided")

    servers = []
    # Check ports and establish SSH tunnels
    for i, url in enumerate(vllm_urls):
        # Extract host from URL
        host = url.split("://")[1].split(":")[0]
        port = int(url.split(":")[-1].split("/")[0])

        # Check if port is in use
        if check_port_in_use(port):
            print(f"Port {port} is in use. Attempting to kill existing process...")
            os.system(f"lsof -ti:{port} | xargs kill -9")

        # Start SSH tunnel using command line
        # Use environment variable or config file to store password
        ssh_password = os.environ.get('SSH_PASSWORD', '')  # Get password from environment variable
        if not ssh_password:
            # Fallback to reading from config file if env var not set
            try:
                with open('config/ssh_config.json') as f:
                    ssh_password = json.load(f).get('password', '')
            except:
                raise ValueError("SSH password not found in environment or config file")
        server = create_ssh_tunnel(host, port, local_port[i], 'hjh', ssh_password, remote_port[i])
        print(f"Established SSH tunnel for {url}")
        servers.append(server)
    return servers

def monitor_tunnel(server):
    """Monitor tunnel connection and restart if needed"""
    while True:
        if not server.is_active:
            try:
                print(f"Tunnel to {server.ssh_host} disconnected, attempting to restart...")
                server.restart()
                print(f"Tunnel restarted successfully")
            except Exception as e:
                print(f"Error restarting tunnel: {e}")
        time.sleep(1)  # Check every 1 seconds


def create_ssh_tunnel(host, remote_port, local_port, username, password, ssh_port):
    tunnel_servers = None
    try:
        tunnel_servers = SSHTunnelForwarder(
            host,
            ssh_username=username,
            ssh_password=password,
            ssh_port=ssh_port,  # 指定 SSH 登录端口
            remote_bind_address=('localhost', remote_port),
            local_bind_address=('localhost', local_port),
            set_keepalive=20
        )
        tunnel_servers.start()
        print(f"SSH tunnel established on local port {local_port}")

        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=monitor_tunnel, args=(tunnel_servers,))
        monitor_thread.daemon = True  # Thread will terminate when main program exits
        monitor_thread.start()

    except Exception as e:
        print(f"Error creating SSH tunnel: {e}")

    return tunnel_servers


def stop_tunnel(server):
    """关闭 SSH 隧道"""
    if server.is_active:
        server.stop()
        print("Tunnel to:" + server.ssh_host + "stopped.")


def some_endpoint_test(base_url):
    """Test the vLLM service by sending a small inference request."""
    url = f"{base_url}/health"
    payload = {
        "prompt": "Test prompt for vLLM",
        "max_tokens": 5
    }
    try:
        response = requests.get(url, timeout=10, json=payload)
        if response.status_code == 200:
            # 健康检查成功，解析返回的 JSON 数据
            return {"status": "success", "data": response.json()}
        else:
            # 健康检查失败，记录错误
            return {
                "status": "failure",
                "error": f"HTTP {response.status_code}",
                "body": response.text
            }
    except requests.exceptions.RequestException as e:
        # 网络或请求异常
        return {"status": "error", "exception": str(e)}