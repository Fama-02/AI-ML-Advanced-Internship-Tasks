from transformers import pipeline
import pandas as pd

# ── 1. Load zero-shot classifier ──────────────────────────────
print("Loading model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("Model loaded!\n")

# ── 2. Define categories ──────────────────────────────────────
labels = ["billing", "technical", "account", "shipping", "refund", "password", "hardware", "software"]

# ── 3. Sample support tickets ─────────────────────────────────
tickets = [
    "My internet connection keeps dropping every hour",
    "I was charged twice for my subscription this month",
    "I cannot login to my account, password not working",
    "My laptop screen is flickering and making noise",
    "I want to cancel my order and get a refund",
    "The app keeps crashing when I open it",
    "I need to update my billing address",
    "My package has not arrived after 2 weeks",
]

# ── 4. Zero-shot classification ───────────────────────────────
print("=" * 60)
print("ZERO-SHOT CLASSIFICATION RESULTS")
print("=" * 60)

results = []
for ticket in tickets:
    result = classifier(ticket, labels, multi_label=False)
    top3 = list(zip(result["labels"][:3], result["scores"][:3]))
    
    print(f"\nTicket: {ticket}")
    print("Top 3 Tags:")
    for i, (label, score) in enumerate(top3, 1):
        print(f"  {i}. {label} ({round(score*100, 1)}%)")
    
    results.append({
        "ticket": ticket,
        "tag1": top3[0][0],
        "tag2": top3[1][0],
        "tag3": top3[2][0],
        "confidence": round(top3[0][1]*100, 1)
    })

# ── 5. Few-shot examples ──────────────────────────────────────
print("\n" + "=" * 60)
print("FEW-SHOT CLASSIFICATION RESULTS")
print("=" * 60)

few_shot_prompt = """Given these examples:
Ticket: "Internet not working" -> Tag: technical
Ticket: "Payment issue" -> Tag: billing  
Ticket: "Cannot login" -> Tag: account

Now classify this ticket: "{ticket}"
The most likely tag is:"""

test_tickets = [
    "My wifi router is not connecting",
    "I was billed incorrectly",
    "I forgot my password",
]

for ticket in test_tickets:
    result = classifier(ticket, labels)
    print(f"\nTicket: {ticket}")
    print(f"Predicted Tag: {result['labels'][0]} ({round(result['scores'][0]*100, 1)}%)")

# ── 6. Save results ───────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print("\n✅ Results saved to results.csv")
print("\nSample Results:")
print(df[["ticket", "tag1", "confidence"]].to_string(index=False))