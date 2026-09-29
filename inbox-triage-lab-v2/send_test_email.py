"""
Self-test harness: sends a simulated phishing-style email from one of your own
accounts to another one you own, so you can verify the detector catches it.

IMPORTANT: Only ever send to email addresses you personally own and control.
These are benign test messages (no real malicious links or payloads) used purely
to validate the detection pipeline end-to-end.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------- Sample simulated phishing templates (benign, no real malicious links) ----------
TEST_TEMPLATES = [
    {
        "subject": "Urgent: Verify your account now",
        "body": (
            "Dear Customer,\n\n"
            "We detected unusual activity on your account. Your account will be suspended "
            "within 24 hours unless you verify your identity immediately.\n\n"
            "Click here to confirm your password and account details: "
            "http://example-test-domain.invalid/verify-account\n\n"
            "Failure to act now will result in permanent suspension.\n\n"
            "Security Team"
        )
    },
    {
        "subject": "You've won a $500 gift card!",
        "body": (
            "Congratulations! You are our lucky winner!\n\n"
            "Claim your $500 gift card now by confirming your bank details at the link below. "
            "This is a limited time offer, act now before it expires!\n\n"
            "http://example-test-domain.invalid/claim-prize\n\n"
            "Prize Team"
        )
    },
    {
        "subject": "Quarterly report attached",
        "body": (
            "Hi,\n\nAttached is the quarterly report we discussed in yesterday's meeting. "
            "Let me know if you have any questions before Friday's review.\n\nThanks,\nTeam"
        )
    },
]


def send_test_email(from_address, from_app_password, to_address, template_index=0):
    template = TEST_TEMPLATES[template_index]

    msg = MIMEMultipart()
    msg["From"] = from_address
    msg["To"] = to_address
    msg["Subject"] = template["subject"]
    msg.attach(MIMEText(template["body"], "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_address, from_app_password)
        server.sendmail(from_address, to_address, msg.as_string())

    print(f"Sent test email '{template['subject']}' to {to_address}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phishing Detector Self-Test")
    print("Only send to email addresses YOU own and control.")
    print("=" * 60)

    from_address = input("Your sending Gmail address: ").strip()
    from_app_password = input("App password for sending account: ").strip()
    to_address = input("Your OWN second email address (test recipient): ").strip()

    print("\nAvailable test templates:")
    for i, t in enumerate(TEST_TEMPLATES):
        print(f"  [{i}] {t['subject']}")

    choice = input("\nWhich template to send? (0-2, or 'all'): ").strip()

    if choice.lower() == "all":
        for i in range(len(TEST_TEMPLATES)):
            send_test_email(from_address, from_app_password, to_address, i)
    else:
        send_test_email(from_address, from_app_password, to_address, int(choice))

    print("\nDone. Now go scan the recipient inbox in your dashboard to see if it was flagged.")
