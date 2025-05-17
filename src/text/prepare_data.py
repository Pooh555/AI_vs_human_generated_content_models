import pandas as pd

# 1. Load original data
df = pd.read_csv("./data/Training_Essay_Data.csv")

# 2. Keep only essays ≥150 words
df["word_count"] = df["text"].str.split().str.len()
filtered = df[df["word_count"] >= 150].drop(columns="word_count")

# 3. Drop exact-duplicate essays
filtered = filtered.drop_duplicates(subset="text").reset_index(drop=True)

# 4. Balance AI vs Human
ai = filtered[filtered["generated"] == 1]
human = filtered[filtered["generated"] == 0]
n = min(len(ai), len(human))
balanced = pd.concat([
    ai.sample(n=n, random_state=42),
    human.sample(n=n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Save your new CSV
balanced.to_csv("./data/Balanced_Essay_Data.csv", index=False)
print(f"Saved {len(balanced)} essays equally split AI/Human.")
