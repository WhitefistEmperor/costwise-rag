"""Builds data/questions.json: the eval question set for the Brightline sample corpus.

Each question has two paraphrases (used as cache repeats), the key facts a correct answer
must contain, the documents that hold the answer, a difficulty label and a split.

"Families" are the same question asked about different plans. They are deliberate near-misses:
almost the same wording, different answers. A semantic cache must not confuse them.

Keywords: a list of groups; every group must appear in the answer. A group is a string or a
list of acceptable alternatives.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "questions.json"

# family name -> (doc, split, three phrasings, {entity: keywords})
FAMILIES = {
    "price": ("pricing.md", "tune",
              ["How much does the {x} plan cost per month?",
               "What is the monthly price of the {x} plan?",
               "What do I pay each month for {x}?"],
              {"Starter": ["9"], "Team": ["49"], "Business": ["199"]}),
    "storage": ("pricing.md", "test",
                ["How much storage is included in the {x} plan?",
                 "What storage allowance comes with {x}?",
                 "How many GB or TB do I get on the {x} plan?"],
                {"Starter": ["100"], "Team": [["1 tb", "1tb", "1 terabyte"]],
                 "Business": [["5 tb", "5tb", "5 terabyte"]]}),
    "backup_freq": ("backups.md", "test",
                    ["How often are backups taken on the {x} plan?",
                     "What is the backup frequency for {x} workspaces?",
                     "On {x}, how frequently does Brightline back up my data?"],
                    {"Starter": [["daily", "every day", "once a day"]],
                     "Team": [["6 hours", "six hours"]],
                     "Business": [["hourly", "every hour"]]}),
    "retention": ("backups.md", "test",
                  ["How long are backups retained on the {x} plan?",
                   "What is the backup retention period for {x}?",
                   "For how many days does {x} keep backups?"],
                  {"Starter": ["30"], "Team": ["90"], "Business": ["365"]}),
    "rate_limit": ("api.md", "test",
                   ["What is the API rate limit on the {x} plan?",
                    "How many API requests per minute does {x} allow?",
                    "What request-per-minute cap applies to {x} API usage?"],
                   {"Starter": ["60"], "Team": ["600"], "Business": ["3000"]}),
    "support_time": ("support.md", "tune",
                     ["What is the support response time on the {x} plan?",
                      "How quickly does support reply to {x} customers?",
                      "How long until I hear back from support on {x}?"],
                     {"Starter": ["48"], "Team": ["8"],
                      "Business": [["1 hour", "one hour", "1-hour"]]}),
    "versions": ("backups.md", "test",
                 ["How many file versions does the {x} plan keep?",
                  "What is the version history limit on {x}?",
                  "How many previous versions of a file are stored on {x}?"],
                 {"Starter": ["10"], "Team": ["50"], "Business": ["unlimited"]}),
    "guests": ("sharing.md", "test",
               ["How many guest collaborators can a {x} workspace invite?",
                "What is the guest limit on the {x} plan?",
                "How many outside guests are allowed on {x}?"],
               {"Team": ["25"], "Business": ["unlimited"]}),
    "extra_seat": ("pricing.md", "tune",
                   ["How much does an extra seat cost on the {x} plan?",
                    "What is the price of an additional seat on {x}?",
                    "What do I pay per extra user on {x}?"],
                   {"Team": ["6"], "Business": ["4"]}),
    "region_fee": ("data-residency.md", "test",
                   ["What does it cost a {x} workspace to change region?",
                    "How much is a region migration on the {x} plan?",
                    "Is there a fee for moving a {x} workspace to another region?"],
                   {"Team": ["150"], "Business": [["free", "no fee", "no charge", "no cost"]]}),
    "refund_window": ("refunds.md", "tune",
                      ["What is the refund window for {x} plans?",
                       "Within how many days can {x} plans be refunded?",
                       "How long do I have to request a refund on {x} subscriptions?"],
                      {"monthly": ["7"], "annual": ["30"]}),
    "mobile_os": ("sync-client.md", "test",
                  ["What is the minimum {x} version for the mobile app?",
                   "Which {x} version does the Brightline mobile app need?",
                   "What {x} release is required to run the mobile app?"],
                  {"iOS": ["16"], "Android": ["11"]}),
}

# (question, [paraphrase, paraphrase], keywords, [docs])
SINGLES = [
    ("How long do refunds take to process?",
     ["How many business days until a refund arrives?", "What is the processing time for an approved refund?"],
     ["5", "10"], ["refunds.md"]),
    ("What encryption is used for data at rest?",
     ["How is stored data encrypted?", "Which cipher protects files at rest?"],
     [["aes-256", "aes 256"]], ["security.md"]),
    ("Which plans support hardware security keys?",
     ["On which plans can I use a hardware key for 2FA?", "Are hardware security keys available on every plan?"],
     ["team", "business"], ["security.md"]),
    ("After how many days do API keys expire?",
     ["How long is an API key valid before it expires?", "What is the lifetime of an API key if I never rotate it?"],
     ["90"], ["api.md"]),
    ("Which languages does support cover?",
     ["In what languages can I contact support?", "What languages are offered by the support team?"],
     ["english", "hindi", "spanish"], ["support.md"]),
    ("Which data regions are available?",
     ["Where can my workspace be hosted?", "What hosting regions does Brightline offer?"],
     ["mumbai", "frankfurt", "virginia"], ["data-residency.md"]),
    ("How long does a region migration take?",
     ["How many hours does changing region take?", "What is the duration of a region change?"],
     ["72"], ["data-residency.md"]),
    ("After how many days do share links expire by default?",
     ["What is the default expiry of a share link?", "How long does a shared link stay active by default?"],
     ["14"], ["sharing.md"]),
    ("How long is the grace period after deleting a workspace?",
     ["For how many days can a deleted workspace be restored?", "What is the restore window after workspace deletion?"],
     ["14"], ["account-deletion.md"]),
    ("What is the maximum file size for the desktop client?",
     ["How large a file can the desktop app sync?", "What is the biggest file the desktop client handles?"],
     ["50"], ["sync-client.md"]),
    ("At what battery level does sync pause?",
     ["When does syncing stop because of low battery?", "Below what battery percentage is sync paused?"],
     ["15"], ["sync-client.md"]),
    ("What GST rate applies to Indian customers?",
     ["How much GST is added for customers in India?", "What tax percentage do Indian customers pay?"],
     ["18"], ["billing.md"]),
    ("How many times is a failed payment retried?",
     ["How many retry attempts follow a failed charge?", "What is the number of retries after a payment fails?"],
     [["3", "three"]], ["billing.md"]),
    ("How long is the free trial?",
     ["How many days does the Team trial last?", "What is the length of the trial period?"],
     ["14"], ["onboarding.md"]),
    ("Up to how much data can be migrated for free?",
     ["What is the free migration allowance?", "How many GB does the migration tool import at no cost?"],
     ["500"], ["onboarding.md"]),
]

HARD = [
    ("What is the total monthly cost of a Team plan with 12 seats and 1.3 TB of storage?",
     ["How much per month would Team cost for 12 users and 1.3 TB?",
      "If I need 12 seats and 1.3 TB on Team, what is my monthly bill?"],
     ["76"], ["pricing.md"]),
    ("How much does the Business plan cost for a full year with annual billing?",
     ["What is the yearly price of Business when paid annually?",
      "If I pay annually, what does one year of Business cost in total?"],
     [["1910.4", "1910.40"]], ["pricing.md"]),
    ("If a payment keeps failing, how many days after the first failure is the workspace suspended?",
     ["Counting from the first failed charge, when does suspension happen?",
      "How long from the first failed payment until the workspace is suspended?"],
     ["30"], ["billing.md"]),
    ("Compare backup frequency and retention between Starter and Business.",
     ["How do Starter and Business differ on backup schedule and how long backups are kept?",
      "What are the backup frequency and retention differences between Starter and Business?"],
     [["daily", "every day"], ["hourly", "every hour"], "30", "365"], ["backups.md"]),
    ("I am on Team and want to move my data from Virginia to Mumbai; what does it cost and how long does it take?",
     ["What are the fee and duration for a Team workspace migrating from Virginia to Mumbai?",
      "Moving a Team workspace to the Mumbai region: how much and how many hours?"],
     ["150", "72"], ["data-residency.md"]),
    ("Which plan do I need for both SSO and unlimited guest collaborators, and what does it cost per month?",
     ["What is the monthly price of the plan that has SAML SSO and unlimited guests?",
      "I want single sign-on plus unlimited guests; which plan and what monthly price?"],
     ["business", "199"], ["security.md", "sharing.md", "pricing.md"]),
    ("What is the cheapest plan with both password-protected share links and hardware security keys, and its monthly price?",
     ["Which is the lowest-priced plan offering hardware keys and password-protected links, and what does it cost monthly?",
      "I need password-protected links and hardware key 2FA; what is the cheapest plan and its price per month?"],
     ["team", "49"], ["sharing.md", "security.md", "pricing.md"]),
    ("How much would it cost to migrate 800 GB from Dropbox?",
     ["What is the migration fee for importing 800 GB?",
      "If I import 800 GB with the migration tool, what do I pay?"],
     ["6"], ["onboarding.md"]),
    ("If I delete my workspace today, what is the maximum number of days until my data is permanently purged?",
     ["After deleting a workspace, how many days at most until everything is purged for good?",
      "What is the longest time between workspace deletion and permanent data purge?"],
     ["44"], ["account-deletion.md"]),
    ("What is the monthly price of the Starter plan including GST for an Indian customer?",
     ["How much does Starter cost per month in India once GST is added?",
      "With GST included, what does an Indian customer pay monthly for Starter?"],
     ["10.62"], ["pricing.md", "billing.md"]),
    ("How many API requests can a Business workspace make in one hour at the rate limit?",
     ["At the maximum rate, how many API calls per hour does Business allow?",
      "What is the hourly API request ceiling for a Business workspace?"],
     ["180000"], ["api.md"]),
    ("Can I sync a 60 GB file with the desktop client, and if not, what is the alternative?",
     ["Is a 60 GB file too large for the desktop app, and how else could I upload it?",
      "What should I do to upload a 60 GB file if the desktop client cannot handle it?"],
     ["multipart"], ["sync-client.md", "api.md"]),
    ("What support channel and response time would I get after upgrading from Starter to Business?",
     ["If I move from Starter to Business, how can I reach support and how fast do they respond?",
      "After upgrading from Starter to Business, what support channel and reply time apply?"],
     ["phone", ["1 hour", "one hour", "1-hour"]], ["support.md"]),
    ("For a 14-person team that needs 90-day backup retention, what is the monthly cost on the cheapest suitable plan?",
     ["What would a team of 14 pay per month on the cheapest plan with 90-day backup retention?",
      "We are 14 people and need backups kept 90 days; what is the lowest monthly cost?"],
     ["73"], ["pricing.md", "backups.md"]),
]


def build() -> list[dict]:
    out: list[dict] = []
    for fam, (doc, split, phr, entities) in FAMILIES.items():
        for ent, kw in entities.items():
            out.append({"id": f"{fam}:{ent.lower()}", "family": fam, "q": phr[0].format(x=ent),
                        "paraphrases": [p.format(x=ent) for p in phr[1:]], "keywords": kw,
                        "docs": [doc], "difficulty": "easy", "split": split})
    for i, (q, paras, kw, docs) in enumerate(SINGLES):
        out.append({"id": f"single:{i:02d}", "family": None, "q": q, "paraphrases": paras, "keywords": kw,
                    "docs": docs, "difficulty": "easy", "split": "tune" if i % 2 == 0 else "test"})
    for i, (q, paras, kw, docs) in enumerate(HARD):
        out.append({"id": f"hard:{i:02d}", "family": None, "q": q, "paraphrases": paras, "keywords": kw,
                    "docs": docs, "difficulty": "hard", "split": "tune" if i % 2 == 0 else "test"})
    return out


if __name__ == "__main__":
    qs = build()
    OUT.write_text(json.dumps(qs, indent=1), encoding="utf-8")
    by = lambda k, v: sum(1 for q in qs if q[k] == v)
    print(f"{len(qs)} questions: {by('difficulty', 'easy')} easy, {by('difficulty', 'hard')} hard; "
          f"{by('split', 'tune')} tune, {by('split', 'test')} test")
