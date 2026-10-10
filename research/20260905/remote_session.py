"""Connect using the workspace connection guide, without logging credentials."""

import argparse
import base64
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for package_dir in (
    ROOT / "SonicRec/.codex_ssh_runtime3",
    ROOT / "SonicRec/.codex_ssh_runtime",
):
    if (package_dir / "paramiko/__init__.py").is_file():
        sys.path.insert(0, str(package_dir))
        break

import paramiko


def read_credentials(host):
    guide = (ROOT / "SERVER_CONNECTION.md").read_text(encoding="utf-8-sig")
    jump = re.search(r"主机[：:]\s*(\S+)\s+端口[：:]\s*(\d+)\s+用户名[：:]\s*(\S+)", guide)
    password = re.search(r"密码[：:]\s*`([^`]+)`", guide)
    accounts = dict(re.findall(r"用户名[：:]\s*`([^`]+)`\s+密码[：:]\s*`([^`]+)`", guide))
    target = re.search(r"ssh\s+([^@\s]+)@211\.81\.248\." + host, guide)
    if not all((jump, password, target)) or "guest" not in accounts:
        raise ValueError("Connection guide does not match the documented format")
    return jump.groups(), password.group(1), accounts, target.group(1)


class Session:
    def __init__(self, host):
        jump, password, accounts, user = read_credentials(host)
        self.secrets = (password, *accounts.values())
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        known = Path(__file__).with_name("known_hosts")
        if known.exists():
            self.client.load_host_keys(str(known))
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(jump[0], port=int(jump[1]), username=jump[2],
                            password=password, timeout=20, auth_timeout=30,
                            banner_timeout=30, look_for_keys=False, allow_agent=False)
        self.client.save_host_keys(str(known))
        self.channel = self.client.invoke_shell(width=180, height=48)
        screen = self.wait_for(r"Type <Enter> for login", 30)
        self.drain(0.5)
        self.channel.send("jj")
        self.drain(0.5)
        self.channel.send("\n")
        screen = self.wait_for(r"<<\s*USERNAME\s*>>|用户名[：:]", 20)
        self.channel.send("guest\n")
        self.wait_for(r"密码|password", 20)
        self.channel.send(accounts["guest"] + "\n")
        self.wait_for(r"[\]~][#$]|guest@|Last login", 30)
        self.channel.send(f"ssh -o StrictHostKeyChecking=accept-new {user}@211.81.248.{host}\n")
        self.wait_for(r"password:|密码", 25)
        self.channel.send(accounts[user] + "\n")
        self.wait_for(re.escape(user) + r"@[^\r\n]+[#$]", 30)
        self.channel.send("stty -echo\n")
        self.drain(0.5)

    def drain(self, duration):
        deadline = time.monotonic() + duration
        chunks = []
        while time.monotonic() < deadline:
            if self.channel.recv_ready():
                chunks.append(self.channel.recv(65536).decode("utf-8", errors="replace"))
            else:
                time.sleep(0.05)
        return "".join(chunks)

    def wait_for(self, pattern, timeout):
        deadline = time.monotonic() + timeout
        text = ""
        while time.monotonic() < deadline:
            text += self.drain(0.2)
            plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
            if re.search(pattern, plain, re.IGNORECASE):
                return plain
        for secret in self.secrets:
            plain = plain.replace(secret, "[REDACTED]")
        raise TimeoutError(f"Login stage {ascii(pattern)}: {ascii(plain[-2500:])}")

    def run_python(self, source, python_bin, timeout):
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        marker = "RESEARCH_DONE_20260905"
        chunks = "\n".join(repr(encoded[i:i + 768]) for i in range(0, len(encoded), 768))
        wrapper = "import base64\nprint('RESEARCH_BEGIN_20260905', flush=True)\nexec(base64.b64decode(\n" + chunks + "\n))"
        import shlex
        command = f"{shlex.quote(python_bin)} -u - <<'RESEARCH_SCRIPT_20260905'\n{wrapper}\nRESEARCH_SCRIPT_20260905\n"
        self.channel.sendall(command + f"printf '\\n{marker}:%s\\n' \"$?\"\n")
        deadline = time.monotonic() + timeout
        output = ""
        while time.monotonic() < deadline:
            output += self.drain(0.25)
            match = re.search(marker + r":(\d+)", output)
            if match:
                result = output[:match.start()]
                result = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result)
                result = result.replace("\r", "")
                start = result.rfind("RESEARCH_BEGIN_20260905\n")
                if start >= 0:
                    result = result[start + len("RESEARCH_BEGIN_20260905\n"):]
                # Redact only nontrivial passwords, preserving ordinary words.
                for secret in self.secrets:
                    if len(secret) >= 10:
                        result = result.replace(secret, "[REDACTED]")
                return int(match.group(1)), result.strip()
        raise TimeoutError("Remote command timeout; inspect server before retrying a launch")

    def close(self):
        self.client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=("213", "214"), default="213")
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--debug-menu", action="store_true")
    parser.add_argument("--save-output", type=Path)
    args = parser.parse_args()
    if args.debug_menu:
        debug_menu(args.host)
        return
    source = args.script.read_text(encoding="utf-8-sig")
    session = Session(args.host)
    try:
        code, output = session.run_python(source, args.python, args.timeout)
        if args.save_output:
            args.save_output.write_text(output + "\n", encoding="utf-8")
        print(output)
        raise SystemExit(code)
    finally:
        session.close()


def debug_menu(host):
    jump, password, accounts, user = read_credentials(host)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(str(Path(__file__).with_name("known_hosts")))
    client.connect(jump[0], port=int(jump[1]), username=jump[2], password=password,
                   timeout=20, look_for_keys=False, allow_agent=False)
    channel = client.invoke_shell(width=120, height=36)
    actions = {"down": "j", "up": "k", "enter": "\r", "lf": "\n",
               "user": "guest", "password": accounts["guest"], "tab": "\t",
               "escape": "\x1b", "one": "1", "select1": ":1\n",
               "target": f"ssh {user}@211.81.248.{host}\n",
               "target_password": accounts[user]}
    try:
        while True:
            deadline = time.monotonic() + 1.5
            data = ""
            while time.monotonic() < deadline:
                if channel.recv_ready():
                    data += channel.recv(65536).decode("utf-8", errors="replace")
                else:
                    time.sleep(0.05)
            for secret in (password, *accounts.values()):
                data = data.replace(secret, "[REDACTED]")
            print(ascii(data), flush=True)
            action = input("ACTION> ").strip()
            if action == "quit":
                break
            if action == "read":
                continue
            if action not in actions:
                print("Unknown action", flush=True)
                continue
            channel.send(actions[action])
    finally:
        client.close()


if __name__ == "__main__":
    main()
