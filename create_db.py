import pandas as pd
import jdatetime
import sqlite3
import re

file_path = "Jabama APM Task Data.xlsx"  # Change if needed
sheet_name = 0  # Or actual sheet name if needed
db_name = "reservations.db"
csv_name = "reservations.csv"

df = pd.read_excel(file_path, sheet_name=sheet_name)

month_map = {
    "Farvardin": 1, "Ordibehesht": 2, "Khordad": 3,
    "Tir": 4, "Mordad": 5, "Shahrivar": 6,
    "Mehr": 7, "Aban": 8, "Azar": 9,
    "Dey": 10, "Bahman": 11, "Esfand": 12
}
def parse_time(value):
    """Parse Persian month date string to jdatetime, return None if invalid."""
    try:
        if pd.isna(value):
            return None

        match = re.match(r"(\w+)\s+(\d+),\s+(\d+),\s+(\d+):(\d+)\s+(AM|PM)", str(value))
        if not match:
            return None

        month_str, day, year, hour, minute, ampm = match.groups()
        day, year, hour, minute = int(day), int(year), int(hour), int(minute)
        month = month_map.get(month_str)
        if not month:
            return None

        if ampm.upper() == "PM" and hour != 12:
            hour += 12
        elif ampm.upper() == "AM" and hour == 12:
            hour = 0

        jdate = jdatetime.datetime(year, month, day, hour, minute)

        gdate = jdate.togregorian()

        weekday_map = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        weekday_name = weekday_map[gdate.weekday()]  # Correct weekday

        return {
            "month": month,
            "day": day,
            "year": year,
            "hour": hour,
            "weekday": weekday_name,
            "jdate": jdate
        }
    except:
        return None

processed_data = []
outliers_count = 0

for _, row in df.iterrows():
    reserve_id = row["ReserveId"]
    submit_parsed = parse_time(row["ReserveSubmitTime"])
    entry_parsed = parse_time(row["Entry"])
    exit_parsed = parse_time(row["Exit"])

    if not submit_parsed or not entry_parsed or not exit_parsed:
        outliers_count += 1
        continue

    residential_days = (exit_parsed["jdate"] - entry_parsed["jdate"]).days

    processed_data.append({
        "ReserveId": reserve_id,
        "ReserveSubmitTime": row["ReserveSubmitTime"],
        "ReserveSubmitMonth": submit_parsed["month"],
        "ReserveSubmitDay": submit_parsed["day"],
        "ReserveSubmitYear": submit_parsed["year"],
        "ReserveSubmitHour": submit_parsed["hour"],
        "ReserveSubmitWeekday": submit_parsed["weekday"],
        "Entry": row["Entry"],
        "EntryMonth": entry_parsed["month"],
        "EntryDay": entry_parsed["day"],
        "EntryYear": entry_parsed["year"],
        "EntryHour": entry_parsed["hour"],
        "EntryWeekday": entry_parsed["weekday"],
        "Exit": row["Exit"],
        "ExitMonth": exit_parsed["month"],
        "ExitDay": exit_parsed["day"],
        "ExitYear": exit_parsed["year"],
        "ExitHour": exit_parsed["hour"],
        "ExitWeekday": exit_parsed["weekday"],
        "VillaId": row["VillaId"],
        "UserId": row["UserId"],
        "ResidentialDays": residential_days
    })

conn = sqlite3.connect(db_name)
pd.DataFrame(processed_data).to_sql("reservations", conn, if_exists="replace", index=False)
conn.close()

pd.DataFrame(processed_data).to_csv(csv_name, index=False, encoding="utf-8-sig")

print(f"✅ Process completed. Ignored {outliers_count} outlier rows.")
print(f"Saved {len(processed_data)} valid rows to DB and CSV.")
