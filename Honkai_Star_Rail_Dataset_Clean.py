# Honkai: Star Rail synthetic player dataset generator

import pandas as pd
import numpy as np 
import random
from datetime import date

# Reproducibility

np.random.seed(42)
random.seed(42)

# 1. CONSTANTS

SNAPSHOT_DATE = date(2026, 9, 20)
LAUNCH_DATE = date (2023, 4, 26)

MAX_ACTIVE_MONTHS = (
    (SNAPSHOT_DATE.year - LAUNCH_DATE.year)*12
    +(SNAPSHOT_DATE.month - LAUNCH_DATE.month)
    + 1
)

USD_PACKS = [0.99, 4.99, 14.99, 29.99, 49.99, 99.99]
USD_PASS = 4.99
USD_BP = [9.99, 19.99]

# 2. COUNTRY METADATA

COUNTRY_META = {
    "China": {
        "server": "China",
        "language": "Chinese",
        "spend_multiplier": 1.10,
        "age_shift": 0.3,   # older: 25-35 is the core group
        "gender_probs": {"Male": 0.70, "Female": 0.29, "Other": 0.01}
    },
    "Japan": {
        "server": "Asia",
        "language": "Japanese",
        "spend_multiplier": 1.20,
        "age_shift": 0.4,   # oldest market: 25-34 largest group, visible 55+ group
        "gender_probs": {"Male": 0.72, "Female": 0.27, "Other": 0.01}
    },
    "United States": {
        "server": "America",
        "language": "English",
        "spend_multiplier": 0.95,
        "age_shift": -0.2,   # younger: 18-24 is the largest group
        "gender_probs": {"Male": 0.57, "Female": 0.42, "Other": 0.01}
    },
    "South Korea": {
        "server": "Asia",
        "language": "Korean",
        "spend_multiplier": 1.10,
        "age_shift": -0.2,   # younger: 18-24 is the largest group
        "gender_probs": {"Male": 0.71, "Female": 0.28, "Other": 0.01}
    },
    "Germany": {
        "server": "Europe",
        "language": "German",
        "spend_multiplier": 0.90,
        "age_shift": 0.2,   # older: German gamers average 39.5 years (all games)
        "gender_probs": {"Male": 0.68, "Female": 0.31, "Other": 0.01}
    },
    "France": {
        "server": "Europe",
        "language": "French",
        "spend_multiplier": 0.90,
        "age_shift": 0.2,   # older: French gamers average 39-40 years (all games)
        "gender_probs": {"Male": 0.67, "Female": 0.32, "Other": 0.01}
    },
    "United Kingdom": {
        "server": "Europe",
        "language": "English",
        "spend_multiplier": 0.95,
        "age_shift": 0.1,   # slightly older: European gamers average 39 years (all games)
        "gender_probs": {"Male": 0.68, "Female": 0.31, "Other": 0.01},
    },
    "Taiwan": {
        "server": "TW/HK/MO",
        "language": "Chinese",
        "spend_multiplier": 1.05,
        "age_shift": 0.0,
        "gender_probs": {"Male": 0.65, "Female": 0.34, "Other": 0.01},
    },
    "Thailand": {
        "server": "Asia",
        "language": "Thai",
        "spend_multiplier": 0.60,
        "age_shift": 0.0,
        "gender_probs": {"Male": 0.68, "Female": 0.31, "Other": 0.01},
    },
    "Others": {
        "server": ["America", "Europe", "Asia", "TW/HK/MO"],
        "language": ["English", "Spanish", "Portuguese",
                     "Indonesian", "Vietnamese", "Russian"],
        "spend_multiplier": 0.65,
        "age_shift": 0.0,
        "gender_probs": {"Male": 0.67, "Female": 0.32, "Other": 0.01},
    }
}

TARGET_REVENUE_SHARE = {
    "China": 0.400, "Japan": 0.217, "United States": 0.130, "South Korea": 0.071,
    "Germany": 0.032, "France": 0.021, "United Kingdom": 0.021, "Taiwan": 0.021,
    "Thailand": 0.011, "Others": 0.076,
}
# Each country's "weight" (its share of players) is calculated here and stored
# in COUNTRY_META, so it is not typed by hand in the dictionary above.
# player share = revenue share / spend multiplier, so that revenue is not counted twice
for country, meta in COUNTRY_META.items():
    meta["weight"] = TARGET_REVENUE_SHARE[country] / meta["spend_multiplier"]
_total_w = sum(meta["weight"] for meta in COUNTRY_META.values())
for meta in COUNTRY_META.values():
    meta["weight"] /= _total_w
    
# 3. DEMOGRAPHICS

AGE_BUCKETS = [
    (12, 15, 0.02),
    (16, 19, 0.18),
    (20, 24, 0.32),
    (25, 29, 0.22),
    (30, 34, 0.14),
    (35, 39, 0.08),
    (40, 49, 0.03),
    (50, 80, 0.01)    
]

# China and Japan have older players than the default, so they get their own
# buckets. Each list covers every age once and its probabilities add up to 1.
AGE_BUCKETS_BY_COUNTRY = {
    "China": [(12, 17, 0.03), (18, 24, 0.20), (25, 35, 0.70), (36, 45, 0.05), (46, 80, 0.02)],
    "Japan": [(12, 15, 0.02), (16, 19, 0.12), (20, 24, 0.22), (25, 34, 0.36), (35, 44, 0.14), (45, 54, 0.08), (55, 80, 0.06)],
}

def _weights_from_buckets(buckets):
    # Turns a list of (lowest age, highest age, probability) buckets into
    # one probability per single age. Each bucket's probability is shared
    # equally between the ages inside it.
    ages, weights = [], []
    for low, high, prob in buckets:
        n = high - low + 1
        for a in range(low, high + 1):
            ages.append(a)
            weights.append(prob / n)
    weight = np.array(weights, dtype=float)
    return np.array(ages), weight / weight.sum()

# Default age table (used by every country except China and Japan)
AGE_VALUES, AGE_WEIGHTS = _weights_from_buckets(AGE_BUCKETS)

# Special age tables for China and Japan
AGE_TABLES = {country: _weights_from_buckets(buckets) for country, buckets in AGE_BUCKETS_BY_COUNTRY.items()}

def generate_age(country):
    # Use the country's own table if it has one, otherwise the default table
    ages, weights = AGE_TABLES.get(country, (AGE_VALUES, AGE_WEIGHTS))
    age = int(np.random.choice(ages, p=weights))

    # age_shift moves every player of the country by that many years:
    # positive = older players, negative = younger players.
    # The values in COUNTRY_META only mark the direction (older or younger
    # market). They are all smaller than 0.5, so they are rounded away below
    # and do not change any age. The older population of China and Japan
    # really comes from their own buckets above.
    # Only a shift of 1 year or more (for example 2.0 or -1.0) moves the ages.
    age = age + COUNTRY_META[country]["age_shift"]

    # keep the result a whole number inside the allowed range 12-80
    return int(np.clip(round(age), 12, 80))

def generate_gender(country): 
    probability = COUNTRY_META [country] ["gender_probs"]
    keys = list(probability.keys())
    p = np.array(list(probability.values()), dtype = float)
    p /= p.sum()
    return str(np.random.choice(keys, p=p))

# 4. TENURE AND ENGAGEMENT

DOWNLOADS_BY_COHORT = [11.13, 8.76, 2.84, 2.71, 2.50, 2.02, 2.65, 1.83, 2.35, 2.10, 2.75,
                        2.50, 2.15, 2.22, 1.99, 2.03, 1.72, 1.45, 1.36, 1.11, 1.33]

DOWNLOADS_AFTER_DATA = 1.31

def generate_active_months():
    months = np.arange(1, MAX_ACTIVE_MONTHS + 1)
    survival = 1.0 / (months ** 0.85)
    cohort_index = MAX_ACTIVE_MONTHS - months
    downloads = np.array([DOWNLOADS_BY_COHORT[i] if i < len(DOWNLOADS_BY_COHORT) else DOWNLOADS_AFTER_DATA for i in cohort_index])
    month_weights = survival * downloads    
    month_weights /= month_weights.sum()
    return int(np.random.choice(months, p=month_weights))

FREQ_CHINA = {1: 0.163, 2: 0.087, 3: 0.100, 4: 0.100, 5: 0.100, 6: 0.103, 7: 0.347}

def generate_session_frequency(active_months, country):
    if active_months <= 3:
        base = {1: 0.06, 2: 0.12, 3: 0.22, 4: 0.25, 5: 0.18, 6: 0.12, 7: 0.05}
    
    elif active_months <= 12:
        base = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.25, 5: 0.20, 6: 0.15, 7: 0.05}
    
    else:
        base = {1: 0.04, 2: 0.08, 3: 0.18, 4: 0.24, 5: 0.22, 6: 0.16, 7: 0.08}

    if country == "China":
        base = {frequency_value: (base[frequency_value] + FREQ_CHINA[frequency_value]) / 2 for frequency_value in base}
        
    keys = np.array(list(base.keys()))
    probability = np.array(list(base.values()), dtype = float)
    probability /= probability.sum()
    return int (np.random.choice(keys, p = probability))

def generate_session_duration(frequency, active_months):
    base = 0.15 + frequency * 0.15
    base += 0.02 * min(active_months, 36)
    base += np.random.normal(0, 0.25)
    return round (float(np.clip(base, 0.10, 3.0)), 2)

# 5. MONTHLY SPENDING LEDGER 

PASS_RENEWAL = 0.85
WHALE_STACK_PROB = 0.40
EVENT_FOMO_BOOST = 1.5


ARCHETYPES = {
    "F2P": {
        "weight": 0.60,
        "p_pass": 0.00, "p_bp": 0.00, "pack_purchase_probability": 0.00,
        "pack_probabilities": None,
    },
    "Low Spenders": {
        "weight": 0.22,
        "p_pass": 0.15, "p_bp": 0.02, "pack_purchase_probability": 0.05,
        # USD_PACKS order: 0.99, 4.99, 14.99, 29.99, 49.99, 99.99
        "pack_probabilities": [0.55, 0.30, 0.10, 0.04, 0.01, 0.00],
    },
    "Dolphin": {
        "weight": 0.14,
        "p_pass": 0.55, "p_bp": 0.35, "pack_purchase_probability": 0.20,
        "pack_probabilities": [0.05, 0.15, 0.25, 0.25, 0.20, 0.10],
    },
    "Whale": {
        "weight": 0.04,
        "p_pass": 0.85, "p_bp": 0.75, "pack_purchase_probability": 0.65,
        "pack_probabilities": [0.00, 0.02, 0.08, 0.20, 0.35, 0.35],
    },
}

def sample_archetype(active_months):
    """
    Latent archetype assignment.
    Conversion effect: longer-tenured players are more likely to have
    converted from F2P to a paying archetype. This mirrors the well-known
    live-service pattern where retention and monetisation correlate, and
    is what makes tenure genuinely predictive of spend in the resulting
    dataset.
    """
    names = list(ARCHETYPES.keys())
    weight = np.array([ARCHETYPES[archetype_name]["weight"] for archetype_name in names], dtype=float)

    # Conversion boost: 1.0 at month 0, ~1.6 at month 42.
    # weight[0] is F2P because "F2P" is the first entry of ARCHETYPES.
    boost = 1.0 + 0.015 * active_months
    weight[0] /= boost          # F2P weight shrinks with tenure
    weight[1:] *= boost         # spender archetypes grow with tenure
    weight = weight / weight.sum()

    archetype_index = int(np.random.choice(len(names), p=weight))
    return names[archetype_index]

def sample_pack(archetype):
    probability = ARCHETYPES[archetype]["pack_probabilities"]
    if probability is None:
        return None
    p = np.array(probability, dtype = float)
    p /= p.sum()
    return float(np.random.choice(USD_PACKS, p = p))

def simulate_spending_ledger(active_months, country, event_participation = "No"):
    archetype = sample_archetype(active_months)
    spend_multiplier = COUNTRY_META[country]["spend_multiplier"]
    archetype_settings = ARCHETYPES[archetype]
    
    purchases = []
    
    burst_months = set()
    if archetype == "Whale" and active_months >= 3 and random.random() < 0.5:
        # pick 3 random months in which this whale spends more than usual
        burst_months = set(random.sample(range(1, active_months + 1), 3))

    # Express Supply Pass works like a subscription:
    #   p_pass  = share of months in which this player owns the pass
    #   renew   = chance to buy it again when the player had it last month
    #   p_start = chance to buy it when the player did NOT have it last month
    #             (chosen so that the long-run share of months is still p_pass)
    p_pass = min(1.0, archetype_settings["p_pass"] * spend_multiplier)
    renew = max(PASS_RENEWAL, p_pass)
    p_start = 1.0 if p_pass >= 1.0 else min (1.0, p_pass * (1 - renew) / (1 - p_pass))
    has_pass = False

    # NOTE: F2P players also go through this loop. Their probabilities are 0,
    # so they never buy anything, but the random numbers are still drawn.
    # Skipping the loop for F2P would change the random sequence and therefore
    # the whole dataset.
    for month in range(1, active_months + 1):
        # Express Supply Pass
        has_pass = random.random() < (renew if has_pass else p_start)
        if has_pass:
            purchases.append((month, "Express Supply Pass", USD_PASS))

        # Nameless Honor (BP)
        if random.random() < min(1.0, archetype_settings["p_bp"] * spend_multiplier):
            purchases.append((month, "Nameless Honor", float(random.choice(USD_BP))))

        # Oneiric Shards (extra packs)
        pack_purchase_probability = archetype_settings["pack_purchase_probability"] * spend_multiplier
        if month in burst_months:
            pack_purchase_probability *= 1.8
            
        if month == active_months and event_participation == "Yes":
            pack_purchase_probability *= EVENT_FOMO_BOOST   
             
        if random.random() < min(1.0, pack_purchase_probability):
            pack = sample_pack(archetype)
            if pack is not None:
                purchases.append((month, "Oneiric Shards", pack))
                # whales often stack a second pack the same month
                if archetype == "Whale":
                    n_extra = np.random.geometric(WHALE_STACK_PROB) - 1
                    if month in burst_months:
                        n_extra *= 2
                    for _ in range(n_extra):
                        purchases.append((month, "Oneiric Shards", sample_pack(archetype)))

    total_expense = round(sum(purchase[2] for purchase in purchases), 2)
    num_purchase = len(purchases)
    last_month_expense = round(sum(purchase[2] for purchase in purchases if purchase[0] == active_months), 2)
    return purchases, num_purchase, total_expense, last_month_expense

def derive_spender_tier(total_expense, active_months): 
    if total_expense == 0:
        return "F2P"
    avg_monthly = total_expense / max(1, active_months)
    if avg_monthly < 5:
        return "Low"
    if avg_monthly < 50:
        return "Dolphin"
    return "Whale"

# 6. PROGRESS AND EVENTS

LOGIT_NOISE_SD = 0.5

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_event_participation(active_months, frequency):
    logit = (-2.0
             + 0.06 * active_months
             + 0.25 * frequency
             + np.random.normal(0, LOGIT_NOISE_SD))
    return "Yes" if random.random() < _sigmoid(logit) else "No"

def generate_progress(active_months, frequency, total_expense, event_participation):
    event_flag = 1.0 if event_participation == "Yes" else 0.0

    story_logit = (-2.0
                   + 0.07 * active_months
                   + 0.20 * frequency
                   + 0.30 * event_flag
                   + 0.0002 * min(total_expense, 3000)
                   + np.random.normal(0, LOGIT_NOISE_SD))
    ach_logit = (-1.5
                 + 0.06 * active_months
                 + 0.25 * frequency
                 + 0.20 * event_flag
                 + 0.0001 * min(total_expense, 3000)
                 + np.random.normal(0, LOGIT_NOISE_SD))

    story = _sigmoid(story_logit)
    achievement_completion_rate   = _sigmoid(ach_logit)
    return round(float(story), 4), round(float(achievement_completion_rate), 4)

# 7. DATASET GENERATION

def generate_player_dataset(n=10000):
    country_names = list(COUNTRY_META.keys())
    country_weights = np.array([COUNTRY_META[c]["weight"] for c in country_names])
    country_weights /= country_weights.sum()

    rows = []
    for _player_index in range(n):
        country = str(np.random.choice(country_names, p=country_weights))
        country_metadata = COUNTRY_META[country]

        gender = generate_gender(country)
        age    = generate_age(country)

        server   = str(random.choice(country_metadata["server"]))   if isinstance(country_metadata["server"], list)   else country_metadata["server"]
        language = str(random.choice(country_metadata["language"])) if isinstance(country_metadata["language"], list) else country_metadata["language"]

        active_months = generate_active_months()
        frequency     = generate_session_frequency(active_months, country)
        duration      = generate_session_duration(frequency, active_months)

        event_participation = generate_event_participation(active_months, frequency)

        _purchases, num_purchases, total_expense, last_month_expense = simulate_spending_ledger(
            active_months, country, event_participation
        )
        spending_tier = derive_spender_tier(total_expense, active_months)

        story, achievement_completion_rate = generate_progress(
            active_months, frequency, total_expense, event_participation
        )

        rows.append({
            "age": age,
            "gender": gender,
            "country": country,
            "server": server,
            "language": language,
            "frequency_sessions_per_week": frequency,
            "avg_session_duration_hours": duration,
            "active_months": active_months,
            "num_purchases": num_purchases,
            "last_month_expense_usd": last_month_expense,
            "total_expense_usd": total_expense,
            "spending_tier": spending_tier,
            "event_participation": event_participation,
            "story_completion_rate": story,
            "achievements_completion_rate": achievement_completion_rate,
        })
    return pd.DataFrame(rows)

# 8. VALIDATION

def _spearman(x, y):
    """Rank correlation without scipy (pandas' method='spearman' needs scipy)."""
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    return float(np.corrcoef(rx, ry)[0, 1])

def validate_dataset(df):
    f2p = df["spending_tier"] == "F2P"
    assert (df.loc[f2p, "total_expense_usd"] == 0).all(), "F2P must have zero spend"
    assert (df.loc[f2p, "num_purchases"] == 0).all(),   "F2P must have zero purchases"
    assert df["avg_session_duration_hours"].between(0.10, 3.0).all()
    assert df["active_months"].max() <= MAX_ACTIVE_MONTHS
    # every country must have at least one player with frequency == 1
    for country in COUNTRY_META:
        country_rows = df[df["country"] == country]
        assert (country_rows["frequency_sessions_per_week"] == 1).any(), \
            f"{country} must allow frequency == 1"
    assert df["total_expense_usd"].ge(0).all()
    assert df["story_completion_rate"].between(0, 1).all()
    assert df["achievements_completion_rate"].between(0, 1).all()

    # --- tenure vs spend --------------------------------------------------
    # total_expense_usd is zero-inflated (~60% F2P) and heavy-tailed
    # (whales). A global Pearson correlation is therefore dominated by
    # the zero mass and understates the real effect. Use three
    # complementary statistics instead.
    c_pearson_all     = df["active_months"].corr(df["total_expense_usd"])
    c_spearman_all    = _spearman(df["active_months"], df["total_expense_usd"])
    spenders          = df[df["total_expense_usd"] > 0]
    c_pearson_spender = spenders["active_months"].corr(spenders["total_expense_usd"])

    # --- other correlations ----------------------------------------------
    c_tenure_story = df["active_months"].corr(df["story_completion_rate"])
    c_tenure_ach   = df["active_months"].corr(df["achievements_completion_rate"])
    c_spend_story  = df["total_expense_usd"].corr(df["story_completion_rate"])
    c_spend_dur    = df["total_expense_usd"].corr(df["avg_session_duration_hours"])

    print(f"corr(tenure, spend)  Pearson  (all rows) = {c_pearson_all:.3f}")
    print(f"corr(tenure, spend)  Spearman (all rows) = {c_spearman_all:.3f}")
    print(f"corr(tenure, spend)  Pearson  (spenders) = {c_pearson_spender:.3f}")
    print(f"corr(tenure, story)                      = {c_tenure_story:.3f}")
    print(f"corr(tenure, achievements)               = {c_tenure_ach:.3f}")
    print(f"corr(spend,  story)                      = {c_spend_story:.3f}")
    print(f"corr(spend,  duration)                   = {c_spend_dur:.3f}")

    print("\nmean total_expense spend by tenure bucket (USD):")
    buckets = pd.cut(
        df["active_months"],
        bins=[0, 3, 6, 12, 24, 100],
        labels=["1-3", "4-6", "7-12", "13-24", "25+"],
    )
    bucket_means = df.groupby(buckets, observed=True)["total_expense_usd"].mean()
    print(bucket_means.round(2))

    print("\nspending_tier distribution:")
    print(df["spending_tier"].value_counts(normalize=True).round(3))
    print("\nmax spend by spending_tier (USD):")
    print(df.groupby("spending_tier")["total_expense_usd"].max().round(2))

    # --- assertions -------------------------------------------------------
    # Spearman is the primary check: it is rank-based, so it is not
    # distorted by the F2P zero mass or by whale outliers. The
    # spenders-only Pearson isolates the tenure effect among the people
    # whose behaviour we actually care about modelling.
    assert c_spearman_all > 0., (
        f"Spearman corr(tenure, spend) too low: {c_spearman_all:.3f}. "
        "Tenure is not driving spend as expected."
    )
    c_spearman_spenders = _spearman(spenders["active_months"], spenders ["total_expense_usd"])
    print(f"corr(tenure, spend | spenders)  Spearman (spenders) too low: {c_spearman_spenders:.3f}.")
    assert c_spearman_spenders > 0.25, (
        f"Spearman corr(tenure, spend | spender) too low: {c_spearman_spenders:.3f}."                      
    )
    
    assert (df["last_month_expense_usd"] <= df["total_expense_usd"] + 1e-9).all(), \
        "Last-month spend cannot exceed lifetime spend"
    
    assert bucket_means.is_monotonic_increasing, (
        "Mean spend must increase monotonically across tenure buckets.\n"
        f"Observed: {bucket_means.round(2).to_dict()}"
    )
    assert c_tenure_story > 0.50, (
        f"corr(tenure, story) too low: {c_tenure_story:.3f}"
    )
    assert c_spend_story >= 0.00, (
        f"Spend must not reduce progress (got {c_spend_story:.3f})."
    )
    assert c_spend_dur >= -0.05, (
        f"Spend must not shorten sessions (got {c_spend_dur:.3f})."
    )

# 9. RUN

if __name__ == "__main__":
    print(f"Generating 10,000 records (max active months = {MAX_ACTIVE_MONTHS})...")
    df = generate_player_dataset(10000)
    validate_dataset(df)

    csv_path  = "honkai_star_rail_player_data.csv"
    xlsx_path = "honkai_star_rail_player_data.xlsx"

    df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV saved: {csv_path}")
    try:
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        print(f"✅ Excel saved: {xlsx_path}")
    except Exception as e:
        print(f"⚠️  Excel export failed: {e}")
        print("   → Install openpyxl with:  pip install openpyxl")
