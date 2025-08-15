import sqlite3
import pandas as pd
import numpy as np
import jdatetime
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import os

DB_PATH = "reservations.db"
TABLE   = "reservations"
OUTDIR  = "analytics_outputs"
PCT_HIGH = 0.95
PCT_LOW  = 0.10
ROLL_WINDOW_DAYS = 14

os.makedirs(OUTDIR, exist_ok=True)

def parse_mixed_str_to_greg(s: str) -> pd.Timestamp:
    """
    ورودی مثل: 'Shahrivar 2, 2022, 12:00 AM'
    ماه جلالی، سال میلادی، ساعت ۱۲ساعته.
    سال جلالی = سال میلادی - ۶۲۱
    """
    if pd.isna(s):
        return pd.NaT
    try:
        parts = [p.strip() for p in str(s).split(",")]
        if len(parts) != 3:
            return pd.NaT
        month_day, gyear_str, time_part = parts
        mname, d_str = month_day.split(" ")
        j_month_map = {
            "Farvardin":1,"Ordibehesht":2,"Khordad":3,"Tir":4,"Mordad":5,"Shahrivar":6,
            "Mehr":7,"Aban":8,"Azar":9,"Dey":10,"Bahman":11,"Esfand":12
        }
        jm = j_month_map.get(mname)
        if jm is None: return pd.NaT
        jd = int(d_str)
        gy = int(gyear_str)

        t = datetime.strptime(time_part, "%I:%M %p")
        hh, mm = t.hour, t.minute

        jy = gy - 621
        jdt = jdatetime.datetime(jy, jm, jd, hh, mm)
        gdt = jdt.togregorian()
        return pd.Timestamp(gdt)
    except Exception:
        return pd.NaT

def to_csv(df: pd.DataFrame, name: str):
    path = os.path.join(OUTDIR, name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(f"SELECT * FROM {TABLE};", conn)
conn.close()

have_parts = all(c in df.columns for c in [
    "ReserveSubmitYear","ReserveSubmitMonth","ReserveSubmitDay","ReserveSubmitHour",
    "EntryYear","EntryMonth","EntryDay","EntryHour",
    "ExitYear","ExitMonth","ExitDay","ExitHour"
])

if have_parts:
    def build_dt_from_parts(y, m, d, h):
        try:
            y = int(y); m = int(m); d = int(d); h = int(h) if pd.notna(h) else 0
            jy = y - 621
            jdt = jdatetime.datetime(jy, m, d, h, 0)
            return pd.Timestamp(jdt.togregorian())
        except Exception:
            return pd.NaT

    df["ReserveSubmit_dt"] = df.apply(lambda r: build_dt_from_parts(r["ReserveSubmitYear"], r["ReserveSubmitMonth"], r["ReserveSubmitDay"], r.get("ReserveSubmitHour", 0)), axis=1)
    df["Entry_dt"]         = df.apply(lambda r: build_dt_from_parts(r["EntryYear"], r["EntryMonth"], r["EntryDay"], r.get("EntryHour", 0)), axis=1)
    df["Exit_dt"]          = df.apply(lambda r: build_dt_from_parts(r["ExitYear"], r["ExitMonth"], r["ExitDay"], r.get("ExitHour", 0)), axis=1)

else:
    df["ReserveSubmit_dt"] = df["ReserveSubmitTime"].apply(parse_mixed_str_to_greg)
    df["Entry_dt"]         = df["Entry"].apply(parse_mixed_str_to_greg)
    df["Exit_dt"]          = df["Exit"].apply(parse_mixed_str_to_greg)

mask_valid = df[["ReserveSubmit_dt","Entry_dt","Exit_dt"]].notna().all(axis=1)
outliers_parse = df.loc[~mask_valid, ["ReserveId","ReserveSubmitTime","Entry","Exit"]].copy()
df = df.loc[mask_valid].copy()


df["StayDays"] = (df["Exit_dt"] - df["Entry_dt"]).dt.days
df["LeadDays"] = (df["Entry_dt"] - df["ReserveSubmit_dt"]).dt.days
df["Week"]     = df["Entry_dt"].dt.to_period("W").apply(lambda p: p.start_time)
df["Month"]    = df["Entry_dt"].dt.to_period("M").apply(lambda p: p.start_time)


villa_counts = df.groupby("VillaId")["ReserveId"].count().reset_index(name="Bookings")
vc_thr = villa_counts["Bookings"].quantile(PCT_HIGH) if len(villa_counts) else np.nan
villa_high = villa_counts[villa_counts["Bookings"] >= vc_thr].sort_values("Bookings", ascending=False) if not np.isnan(vc_thr) else villa_counts.iloc[0:0]

top10_villas = villa_counts.sort_values("Bookings", ascending=False).head(10)
if not top10_villas.empty:
    plt.figure()
    plt.bar(top10_villas["VillaId"].astype(str), top10_villas["Bookings"])
    plt.title("Top 10 Villas by Bookings")
    plt.xlabel("VillaId"); plt.ylabel("Bookings")
    save_plot(os.path.join(OUTDIR, "top10_villas_bookings.png"))


user_counts = df.groupby("UserId")["ReserveId"].count().reset_index(name="Bookings")
uc_thr = user_counts["Bookings"].quantile(PCT_HIGH) if len(user_counts) else np.nan
user_high = user_counts[user_counts["Bookings"] >= uc_thr].sort_values("Bookings", ascending=False) if not np.isnan(uc_thr) else user_counts.iloc[0:0]
top_user = user_counts.loc[user_counts["Bookings"].idxmax()]
print(top_user["UserId"], top_user["Bookings"])

plt.figure()
plt.hist(user_counts["Bookings"].values, bins=20)
plt.title("Distribution: Bookings per User")
plt.xlabel("Bookings"); plt.ylabel("Users")
save_plot(os.path.join(OUTDIR, "hist_bookings_per_user.png"))

user_counts_filtered = user_counts[user_counts['Bookings'] <= 50]

plt.figure(figsize=(8, 5))
plt.hist(user_counts_filtered['Bookings'], bins=50)
plt.xticks(range(0, 51, 2))
plt.title('Distribution: Bookings per User (Bookings ≤ 50)')
plt.xlabel('Bookings')
plt.ylabel('Users')
save_plot(os.path.join(OUTDIR, "bookings_distribution_under_50.png"))
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(user_counts['Bookings'], bins=50)
plt.yscale('log')
plt.title('Distribution: Bookings per User (Log Scale)')
plt.xlabel('Bookings')
plt.ylabel('Users (log scale)')
save_plot(os.path.join(OUTDIR, "bookings_distribution_log_scale.png"))
plt.show()

overlaps_user = []
for uid, g in df.sort_values("Entry_dt").groupby("UserId"):
    rows = g[["VillaId","Entry_dt","Exit_dt","ReserveId"]].sort_values("Entry_dt").values.tolist()
    for i in range(len(rows)):
        v1, s1, e1, r1 = rows[i]
        for j in range(i+1, len(rows)):
            v2, s2, e2, r2 = rows[j]
            if s2 < e1:
                overlaps_user.append((uid, r1, v1, s1, e1, r2, v2, s2, e2))
            else:
                break
overlap_user_df = pd.DataFrame(overlaps_user, columns=[
    "UserId","ReserveId_1","VillaId_1","Entry_1","Exit_1","ReserveId_2","VillaId_2","Entry_2","Exit_2"
])


overlaps_villa = []
for vid, g in df.sort_values("Entry_dt").groupby("VillaId"):
    rows = g[["ReserveId","Entry_dt","Exit_dt","UserId"]].sort_values("Entry_dt").values.tolist()
    for i in range(len(rows)):
        r1, s1, e1, u1 = rows[i]
        for j in range(i+1, len(rows)):
            r2, s2, e2, u2 = rows[j]
            if s2 < e1:
                overlaps_villa.append((vid, r1, u1, s1, e1, r2, u2, s2, e2))
            else:
                break
overlap_villa_df = pd.DataFrame(overlaps_villa, columns=[
    "VillaId","ReserveId_1","UserId_1","Entry_1","Exit_1","ReserveId_2","UserId_2","Entry_2","Exit_2"
])


villa_avg_stay = df.groupby("VillaId")["StayDays"].mean().reset_index(name="AvgStayDays")
vas_thr = villa_avg_stay["AvgStayDays"].quantile(PCT_HIGH) if len(villa_avg_stay) else np.nan
villa_long_stay = villa_avg_stay[villa_avg_stay["AvgStayDays"] >= vas_thr].sort_values("AvgStayDays", ascending=False) if not np.isnan(vas_thr) else villa_avg_stay.iloc[0:0]


user_villa_counts = df.groupby(["UserId","VillaId"])["ReserveId"].count().reset_index(name="TimesBooked")
loyal_pairs = user_villa_counts[user_villa_counts["TimesBooked"] >= 3].sort_values("TimesBooked", ascending=False)


user_distinct_villas = df.groupby("UserId")["VillaId"].nunique().reset_index(name="DistinctVillas")
udv_thr = user_distinct_villas["DistinctVillas"].quantile(PCT_HIGH) if len(user_distinct_villas) else np.nan
user_explorers = user_distinct_villas[user_distinct_villas["DistinctVillas"] >= udv_thr].sort_values("DistinctVillas", ascending=False) if not np.isnan(udv_thr) else user_distinct_villas.iloc[0:0]


lead = df[["ReserveId","UserId","VillaId","ReserveSubmit_dt","Entry_dt","LeadDays"]].copy()
lead_valid = lead[lead["LeadDays"].notna()].copy()
ld_hi = lead_valid["LeadDays"].quantile(PCT_HIGH) if not lead_valid.empty else np.nan
ld_lo = lead_valid["LeadDays"].quantile(PCT_LOW)  if not lead_valid.empty else np.nan
planners = lead_valid[lead_valid["LeadDays"] >= ld_hi].sort_values("LeadDays", ascending=False) if not np.isnan(ld_hi) else lead_valid.iloc[0:0]
last_minute = lead_valid[lead_valid["LeadDays"] <= ld_lo].sort_values("LeadDays", ascending=True) if not np.isnan(ld_lo) else lead_valid.iloc[0:0]

plt.figure()
plt.hist(lead_valid["LeadDays"], bins=30)
plt.title("Lead Time (days) Histogram")
plt.xlabel("Days between ReserveSubmit and Entry"); plt.ylabel("Reservations")
save_plot(os.path.join(OUTDIR, "hist_lead_time.png"))


g_counts = (
    df.set_index("Entry_dt")
    .groupby("VillaId")["ReserveId"]
    .resample("D").count()
    .rename("DailyBookings")
    .reset_index()
)
g_counts["Rolling14"] = g_counts.groupby("VillaId")["DailyBookings"].transform(lambda s: s.rolling(ROLL_WINDOW_DAYS, min_periods=1).sum())
def flag_bursts(sub):
    thr = sub["Rolling14"].quantile(PCT_HIGH) if len(sub) else np.nan
    sub["IsBurst"] = sub["Rolling14"] >= thr if not np.isnan(thr) else False
    sub["BurstThreshold"] = thr
    return sub
villa_bursts = g_counts.groupby("VillaId", group_keys=False).apply(flag_bursts)
villa_burst_days = villa_bursts[villa_bursts["IsBurst"]]

if not top10_villas.empty:
    top_villa_id = int(top10_villas.iloc[0]["VillaId"])
    vser = villa_bursts[villa_bursts["VillaId"] == top_villa_id]
    if not vser.empty:
        plt.figure()
        plt.plot(vser["Entry_dt"], vser["Rolling14"])
        plt.title(f"Rolling {ROLL_WINDOW_DAYS}-day Bookings - Villa {top_villa_id}")
        plt.xlabel("Date"); plt.ylabel(f"Bookings (rolling {ROLL_WINDOW_DAYS}d)")
        save_plot(os.path.join(OUTDIR, "rolling14_top_villa.png"))


span = df.groupby("VillaId").agg(
    FirstEntry=("Entry_dt","min"),
    LastExit=("Exit_dt","max"),
    TotalBookings=("ReserveId","count")
).reset_index()
span["ActiveDays"] = (span["LastExit"] - span["FirstEntry"]).dt.days
span_thr_book = span["TotalBookings"].quantile(PCT_HIGH) if not span.empty else np.nan
span_thr_days  = span["ActiveDays"].quantile(PCT_HIGH) if not span.empty else np.nan
villa_long_term_popular = span[(span["TotalBookings"] >= span_thr_book) & (span["ActiveDays"] >= span_thr_days)] if not (np.isnan(span_thr_book) or np.isnan(span_thr_days)) else span.iloc[0:0]


weekly = df.set_index("Entry_dt").resample("W")["ReserveId"].count().reset_index(name="Bookings")
plt.figure()
plt.plot(weekly["Entry_dt"], weekly["Bookings"])
plt.title("Weekly Bookings (All Villas)")
plt.xlabel("Week"); plt.ylabel("Bookings")
save_plot(os.path.join(OUTDIR, "weekly_bookings.png"))

weekday_counter = Counter()

for _, row in df.iterrows():
    if pd.isna(row["Entry_dt"]) or pd.isna(row["Exit_dt"]):
        continue
    current_day = row["Entry_dt"].normalize()
    last_day = row["Exit_dt"].normalize()
    while current_day <= last_day:
        weekday_counter[current_day.weekday()] += 1
        current_day += pd.Timedelta(days=1)

weekday_df = pd.DataFrame.from_dict(weekday_counter, orient="index", columns=["Count"]).sort_index()

weekday_map = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
               4: "Thursday", 5: "Friday", 6: "Saturday"}
weekday_df["Weekday"] = weekday_df.index.map(weekday_map)

plt.figure(figsize=(6, 6))
plt.pie(weekday_df["Count"], labels=weekday_df["Weekday"], autopct='%1.1f%%', startangle=90)
plt.title("Reserved Days of Week")
save_plot(os.path.join(OUTDIR, "reserved_days_pie.png"))


from scipy.stats import spearmanr
corr_villa = spearmanr(-villa_counts["VillaId"], villa_counts["Bookings"]) if len(villa_counts) > 1 else (np.nan, np.nan)
user_counts_ren = user_counts.rename(columns={"Bookings":"TotalBookings"})
user_first_entry = df.groupby("UserId")["Entry_dt"].min().reset_index(name="FirstEntry")
user_metrics = user_first_entry.merge(user_counts_ren, on="UserId", how="left")
corr_user = spearmanr(-user_metrics["UserId"], user_metrics["TotalBookings"]) if len(user_metrics) > 1 else (np.nan, np.nan)


paths = {}
def add(df_, name): paths[name] = to_csv(df_, name)

add(villa_counts, "villa_counts.csv")
add(villa_high, "villa_high_outliers.csv")
add(user_counts, "user_counts.csv")
add(user_high, "user_high_outliers.csv")
add(overlap_user_df, "overlap_user.csv")
add(overlap_villa_df, "overlap_villa.csv")
add(villa_avg_stay, "villa_avg_stay.csv")
add(villa_long_stay, "villa_long_stay_outliers.csv")
add(user_villa_counts, "user_villa_loyal_pairs.csv")
add(user_distinct_villas, "user_distinct_villas.csv")
add(user_explorers, "user_explorers_outliers.csv")
add(planners, "lead_planners.csv")
add(last_minute, "lead_last_minute.csv")
add(villa_burst_days, "villa_burst_days.csv")
add(villa_long_term_popular, "villa_long_term_popular.csv")
add(weekly, "weekly_bookings.csv")
add(span, "span_villa_activity.csv")
add(outliers_parse, "outliers_parsing.csv")

print("==== SUMMARY ====")
print(f"Parsed outliers (invalid dates): {len(outliers_parse)} rows")
print(f"Spearman (VillaId vs TotalBookings): rho={corr_villa[0]:.3f}, p={corr_villa[1]:.3g}")
print(f"Spearman (UserId vs TotalBookings): rho={corr_user[0]:.3f}, p={corr_user[1]:.3g}")