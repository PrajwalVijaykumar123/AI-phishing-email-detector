"""
Phishing Email Lab v2 - Flask backend
Connects to Gmail via IMAP, classifies emails using the trained LSTM model,
extracts links/attachments, checks sender domain mismatches, and generates
actionable "don't do this" warnings. Stores results in SQLite, serves a dashboard.
"""

import imaplib
import email
from email.header import decode_header
import re
import pickle
import sqlite3
import json
from datetime import datetime

from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

MAX_LEN = 200
DB_PATH = "lab.db"

print("Loading model...")
model = load_model("phishing_lstm_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Model loaded.")

SUSPICIOUS_TERMS = {
    "urgent": "urgency", "act now": "urgency", "immediately": "urgency",
    "limited time": "urgency", "expire": "urgency",
    "verify": "credential_harvesting", "confirm": "credential_harvesting",
    "password": "credential_harvesting", "login": "credential_harvesting",
    "credentials": "credential_harvesting", "update your": "credential_harvesting",
    "suspend": "account_threat", "account": "account_threat",
    "unusual activity": "account_threat", "security alert": "account_threat",
    "click here": "malicious_link",
    "bank": "financial", "wire transfer": "financial", "gift card": "financial",
    "social security": "financial",
    "winner": "social_engineering", "prize": "social_engineering",
}

BRAND_DOMAINS = {
    "paypal": ["paypal.com"],
    "amazon": ["amazon.com"],
    "apple": ["apple.com", "icloud.com"],
    "microsoft": ["microsoft.com", "outlook.com", "live.com"],
    "google": ["google.com", "gmail.com"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com", "fb.com"],
    "instagram": ["instagram.com"],
    "linkedin": ["linkedin.com"],
    "dropbox": ["dropbox.com"],
    "docusign": ["docusign.com", "docusign.net"],
    "irs": ["irs.gov"],
    "fedex": ["fedex.com"],
    "ups": ["ups.com"],
    "dhl": ["dhl.com"],
    "chase": ["chase.com"],
    "wells fargo": ["wellsfargo.com"],
    "bank of america": ["bankofamerica.com"],
}

URL_REGEX = re.compile(r'(https?://[^\s<>"\']+)', re.IGNORECASE)

TRUSTED_DOMAINS = [
    "google.com", "accounts.google.com", "apple.com", "microsoft.com",
    "github.com", "linkedin.com",
]

def check_sender_authentication(msg):
    headers = msg.get_all("Authentication-Results", []) + msg.get_all("ARC-Authentication-Results", [])
    combined = " ".join(headers).lower()
    spf_pass = "spf=pass" in combined
    dkim_pass = "dkim=pass" in combined
    return {"spf_pass": spf_pass, "dkim_pass": dkim_pass}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            subject TEXT,
            body TEXT,
            received_at TEXT,
            scanned_at TEXT,
            phishing_probability REAL,
            severity TEXT,
            suspicious_terms TEXT,
            links TEXT,
            attachments TEXT,
            warnings TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_suspicious_terms(original_text):
    text_lower = original_text.lower()
    found = {}
    for term, category in SUSPICIOUS_TERMS.items():
        if term in text_lower:
            found[term] = category
    return found

def extract_links(text):
    return list(set(URL_REGEX.findall(text)))

def extract_sender_domain(sender):
    match = re.search(r'@([\w\.-]+)', sender)
    return match.group(1).lower() if match else ""

def check_brand_mismatch(sender, text):
    sender_domain = extract_sender_domain(sender)
    text_lower = text.lower()
    mismatches = []
    for brand, legit_domains in BRAND_DOMAINS.items():
        if brand in text_lower:
            if not any(ld in sender_domain for ld in legit_domains):
                mismatches.append({"brand": brand, "sender_domain": sender_domain})
    return mismatches

def severity_from_probability(prob):
    return "Suspicious" if prob >= 0.5 else "Safe"

def build_warnings(prob, links, attachments, brand_mismatches, suspicious_terms,
                    sender_domain="", auth=None):
    warnings = []
    auth = auth or {"spf_pass": False, "dkim_pass": False}

    domain_is_trusted = any(td in sender_domain for td in TRUSTED_DOMAINS)
    is_verified = domain_is_trusted and auth["spf_pass"] and auth["dkim_pass"]

    if prob >= 0.5 and is_verified:
        warnings.append(
            f"This email matched some phishing-style language, but it passed SPF/DKIM "
            f"authentication from a trusted domain ({sender_domain}). Likely a legitimate "
            f"automated notification, not phishing. Still avoid clicking links you weren't expecting."
        )
        return warnings

    if prob >= 0.5:
        if domain_is_trusted and not (auth["spf_pass"] and auth["dkim_pass"]):
            warnings.append(
                f"This email claims to be from a trusted domain ({sender_domain}) but did NOT "
                f"pass SPF/DKIM authentication. This is a strong sign of a spoofed sender — "
                f"treat with extra caution."
            )
        for link in links:
            warnings.append(f"Don't click this link: {link}")

        for att in attachments:
            warnings.append(f"Don't open this attachment: {att}")

        for m in brand_mismatches:
            warnings.append(
                f"This email claims to be from {m['brand'].title()}, but was sent from "
                f"'{m['sender_domain']}', which doesn't match {m['brand'].title()}'s real domain. "
                f"Treat this as impersonation."
            )

        if any(cat == "credential_harvesting" for cat in suspicious_terms.values()):
            warnings.append("Never enter your password or login details through a link in an email.")

        if any(cat == "financial" for cat in suspicious_terms.values()):
            warnings.append("Don't share bank details, gift card codes, or wire transfer info based on this email.")

        if any(cat == "urgency" for cat in suspicious_terms.values()):
            warnings.append("This email uses urgency to rush you — slow down and verify independently before acting.")

        if not warnings:
            warnings.append("This email matches phishing language patterns. Review carefully before taking any action.")
    else:
        warnings.append("No specific threats detected. Still, always verify unexpected requests independently.")

    return warnings

def classify_email(sender, subject, body):
    full_text = f"{subject} {body}"
    cleaned = clean_text(full_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])

    found_terms = find_suspicious_terms(full_text)
    links = extract_links(body)
    brand_mismatches = check_brand_mismatch(sender, full_text)
    severity = severity_from_probability(prob)

    return {
        "phishing_probability": round(prob * 100, 2),
        "severity": severity,
        "suspicious_terms": list(found_terms.keys()),
        "links": links,
        "brand_mismatches": brand_mismatches,
        "found_terms_categories": found_terms,
    }

def decode_mime_words(s):
    if not s:
        return ""
    decoded = decode_header(s)
    return "".join(
        part.decode(enc or "utf-8", errors="ignore") if isinstance(part, bytes) else part
        for part, enc in decoded
    )

def get_email_body_and_attachments(msg):
    body = ""
    attachments = []
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")
            filename = part.get_filename()

            if filename and "attachment" in disposition:
                attachments.append(decode_mime_words(filename))
            elif content_type == "text/plain" and "attachment" not in disposition:
                try:
                    body = part.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    continue
    else:
        try:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        except Exception:
            body = ""
    return body, attachments

def fetch_and_scan(gmail_user, gmail_app_password, num_emails=15):
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(gmail_user, gmail_app_password)
    imap.select("INBOX")

    status, messages = imap.search(None, "ALL")
    email_ids = messages[0].split()
    email_ids = email_ids[-num_emails:]

    conn = sqlite3.connect(DB_PATH)
    new_count = 0

    for eid in reversed(email_ids):
        status, msg_data = imap.fetch(eid, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject = decode_mime_words(msg.get("Subject"))
        sender = decode_mime_words(msg.get("From"))
        date_str = msg.get("Date", "")
        body, attachments = get_email_body_and_attachments(msg)
        body = body[:3000]

        existing = conn.execute(
            "SELECT id FROM emails WHERE sender=? AND subject=? AND received_at=?",
            (sender, subject, date_str)
        ).fetchone()
        if existing:
            continue

        result = classify_email(sender, subject, body)
        auth_result = check_sender_authentication(msg)
        sender_domain = extract_sender_domain(sender)
        warnings = build_warnings(
            result["phishing_probability"] / 100,
            result["links"],
            attachments,
            result["brand_mismatches"],
            result["found_terms_categories"],
            sender_domain=sender_domain,
            auth=auth_result,
        )

        conn.execute("""
            INSERT INTO emails (sender, subject, body, received_at, scanned_at,
                                 phishing_probability, severity, suspicious_terms,
                                 links, attachments, warnings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sender, subject, body, date_str, datetime.now().isoformat(),
            result["phishing_probability"], result["severity"],
            ", ".join(result["suspicious_terms"]),
            json.dumps(result["links"]),
            json.dumps(attachments),
            json.dumps(warnings),
        ))
        new_count += 1

    conn.commit()
    conn.close()
    imap.logout()
    return new_count

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/emails")
def api_emails():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM emails ORDER BY scanned_at DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.json
    gmail_user = data.get("gmail_user")
    gmail_app_password = data.get("gmail_app_password")
    num_emails = int(data.get("num_emails", 15))

    if not gmail_user or not gmail_app_password:
        return jsonify({"error": "Missing credentials"}), 400

    try:
        new_count = fetch_and_scan(gmail_user, gmail_app_password, num_emails)
        return jsonify({"success": True, "new_emails_scanned": new_count})
    except imaplib.IMAP4.error as e:
        return jsonify({"error": f"IMAP login failed: {str(e)}"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats")
def api_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
    suspicious = conn.execute("SELECT COUNT(*) FROM emails WHERE severity='Suspicious'").fetchone()[0]
    safe = conn.execute("SELECT COUNT(*) FROM emails WHERE severity='Safe'").fetchone()[0]
    conn.close()
    return jsonify({"total": total, "suspicious": suspicious, "safe": safe})

if __name__ == "__main__":
    app.run(debug=True, port=5050)
