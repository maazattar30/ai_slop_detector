"""
Run this locally to generate your ADMIN_PASSWORD_HASH for HF Secrets.
Usage: python gen_password_hash.py
"""
import hashlib

password = input("Enter your admin password: ").strip()
h = hashlib.sha256(password.encode()).hexdigest()
print(f"\nYour ADMIN_PASSWORD_HASH:\n{h}\n")
print("Paste this into HF Space → Settings → Variables and secrets → ADMIN_PASSWORD_HASH")
