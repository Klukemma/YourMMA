"""Guardrails for what agents may run and fetch.

Two layers sit in front of every shell command and URL fetch:

1. A denylist of command shapes that are destructive or classic malware moves
   (privilege escalation, wiping disks, reverse shells, crypto miners, piping a
   download from an unknown site straight into a shell).
2. A domain allowlist: any URL that appears in a command, or that an agent
   asks to fetch as raw content, must belong to a trusted host (official
   package registries, GitHub, language vendors). Anything else is refused.

This is not a perfect sandbox (a determined program can always build a URL at
runtime), which is why the whole thing also runs inside a throwaway container
as an unprivileged user with no access to the app's secrets. The checks here
stop the realistic failure: a model copy-pasting an install line from a sketchy
blog post.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

# Hosts (and all their subdomains) agents may download from.
DEFAULT_TRUSTED_DOMAINS = [
    # Python
    "pypi.org", "pythonhosted.org", "python.org",
    # JavaScript
    "npmjs.org", "npmjs.com", "yarnpkg.com", "nodejs.org", "deno.land", "bun.sh",
    "jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com",
    # Source hosting
    "github.com", "githubusercontent.com", "gitlab.com", "bitbucket.org",
    # Other language ecosystems
    "crates.io", "rust-lang.org", "rustup.rs", "go.dev", "golang.org",
    "proxy.golang.org", "sum.golang.org", "rubygems.org", "maven.org",
    "gradle.org", "nuget.org", "packagist.org", "hex.pm", "pub.dev",
    # OS packages
    "debian.org", "ubuntu.com",
    # Models and data
    "huggingface.co", "hf.co", "kaggle.com",
]

LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

_DENY: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(^|[\s;&|(`])sudo\b"), "sudo is not available to agents"),
    (re.compile(r"(^|[\s;&|(`])su(\s+-|\s+root|\s*$)"), "switching users is not allowed"),
    (re.compile(r"\brm\s+(-\S+\s+)*(/|/\*|~|~/|\$HOME|\$\{HOME\})(\s|$|;)"),
     "refusing to delete the root or home directory"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "formatting filesystems is not allowed"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/"), "writing to raw devices is not allowed"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|vd|xvd)"), "writing to raw devices is not allowed"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "fork bomb"),
    (re.compile(r"\b(shutdown|reboot|poweroff|halt|init\s+0)\b"), "power commands are not allowed"),
    (re.compile(r"\b(systemctl|crontab|iptables|nft|modprobe|insmod)\b"),
     "system administration commands are not allowed"),
    (re.compile(r"\bchmod\s+(-\S+\s+)*[ugoa]*\+s\b|\bchmod\s+(-\S+\s+)*[0-7]?[4-7][0-7]{3}\b"),
     "setuid/setgid permissions are not allowed"),
    (re.compile(r"/dev/(tcp|udp)/"), "raw socket redirection (reverse shell pattern)"),
    (re.compile(r"\b(nc|ncat|netcat)\b[^\n]*\s-(e|c)\b"), "netcat exec (reverse shell pattern)"),
    (re.compile(r"\b(socat)\b[^\n]*\bexec:", re.I), "socat exec (reverse shell pattern)"),
    (re.compile(r"\b(xmrig|minerd|cpuminer|cgminer|ethminer|nbminer|t-rex)\b|stratum\+tcp", re.I),
     "crypto miners are not allowed"),
    (re.compile(r"base64\s+(-d|--decode)[^\n]*\|\s*(ba|z|da)?sh\b"),
     "executing base64-decoded payloads is not allowed"),
    (re.compile(r"/proc/(\d+|self|\*)/(environ|mem)\b"), "reading process memory/environment is not allowed"),
    (re.compile(r"169\.254\.169\.254|metadata\.google\.internal"), "cloud metadata endpoints are off limits"),
    (re.compile(r"\bhistory\s+-c\b|\bshred\b"), "covering tracks is not allowed"),
]

# A download piped straight into an interpreter.
_PIPE_TO_SHELL = re.compile(r"\b(curl|wget)\b[^\n;&]*\|\s*(sudo\s+)?(ba|z|da|k)?sh\b|\b(curl|wget)\b[^\n;&]*\|\s*python")

_URL = re.compile(r"""(?:https?|ftp|git|ssh)://[^\s'"`<>)\]]+""", re.I)
_SCP_GIT = re.compile(r"\b[\w.-]+@([\w.-]+):[\w./-]+")  # git@github.com:owner/repo


def host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def is_trusted_host(host: str, trusted: list[str]) -> bool:
    host = host.lower().rstrip(".")
    if host in LOCAL_HOSTS:
        return True
    return any(host == d or host.endswith("." + d) for d in trusted)


def check_command(command: str, trusted: list[str] | None = None) -> tuple[bool, str]:
    """Return (allowed, reason). reason is empty when allowed."""
    trusted = trusted if trusted is not None else DEFAULT_TRUSTED_DOMAINS
    for pattern, reason in _DENY:
        if pattern.search(command):
            return False, reason

    hosts = {host_of(u) for u in _URL.findall(command)}
    hosts |= {m.lower() for m in _SCP_GIT.findall(command)}
    hosts.discard("")
    untrusted = sorted(h for h in hosts if not is_trusted_host(h, trusted))
    if untrusted:
        return False, (
            "downloads/connections are only allowed to trusted domains; not trusted: "
            + ", ".join(untrusted)
            + ". Use an official package registry or GitHub instead."
        )

    if _PIPE_TO_SHELL.search(command) and not hosts:
        return False, "piping a download into a shell requires an explicit trusted URL"
    return True, ""


def is_public_address(host: str) -> bool:
    """False for private, loopback, link-local and other internal addresses (SSRF guard)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def check_fetch_url(url: str) -> tuple[bool, str]:
    """Reading web pages is allowed anywhere public; internal addresses are not."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, "only http(s) URLs can be fetched"
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL has no host"
    if not is_public_address(host):
        return False, "internal or unresolvable addresses cannot be fetched"
    return True, ""
