"""India 2047 generation model. All quantities are GENERATION in TWh (no capacity, no consumption).
Run: python src/model.py   -> prints scenarios, tornado, 2047 mix; writes outputs.csv"""
import csv

# --- Inputs (source: MNRE RE Statistics 2024-25, Tables 9.1/9.2, from CEA) ---
BASE_YEAR, TARGET_YEAR = 2025, 2047
PC0 = 1304.0          # kWh generated per person, 2024-25 (1,824.1 TWh / 1.399 bn)
POP47 = 1.6375        # bn people, 2047 (CSV-based population forecast)
G_BASE = 0.04026      # per-capita growth/yr (FY15-FY25 trend; rounds to 4.0%)
U_NOW, U_BASE = 37.0, 51.0   # urban share %, now and base 2047 (UN/MoHFW range; verify)
GAMMA = 1.8           # ASSUMPTION: pts of growth per 1 pt/yr change in urbanisation pace
ZONE_NOW = {"West": 458.8, "South": 402.3, "Central": 336.4, "North": 328.9, "East": 274.3, "North-East": 23.4}
ZONE_SHARE_47 = {"West": 1306, "South": 1081, "Central": 753, "North": 1087, "East": 771, "North-East": 89}
MIX_SHARES = {"thermal": .286, "gas_in_thermal": .030, "nuclear": .07, "wind": .11, "hydro": .057, "other": .007}
YEARS = TARGET_YEAR - BASE_YEAR

def growth(econ_adj=0.0, urban47=U_BASE):
    """Per-capita growth = base trend + economy/electrification adj + urbanisation effect (all in fraction/yr)."""
    pace, base_pace = (urban47 - U_NOW) / YEARS, (U_BASE - U_NOW) / YEARS
    return G_BASE + econ_adj / 100 + GAMMA * (pace - base_pace) / 100

def total_2047(g=None, pop=POP47):
    g = growth() if g is None else g
    return PC0 * (1 + g) ** YEARS * pop      # kWh/person x bn people = TWh

def mix(total, s=MIX_SHARES):
    m = {k: total * v for k, v in s.items() if k != "gas_in_thermal"}
    m["coal"], m["gas"] = m["thermal"] - total * s["gas_in_thermal"], total * s["gas_in_thermal"]
    m["solar"] = total - sum(m[k] for k in ("thermal", "nuclear", "wind", "hydro", "other"))  # balancing source
    return m

def zones(total):
    t = sum(ZONE_SHARE_47.values())
    return {z: total * v / t for z, v in ZONE_SHARE_47.items()}

def tornado(total0):
    """Change in 2047 total (TWh) for each driver moved to its low/high value."""
    cases = {"Per-capita growth ±1.2 pts": (lambda: total_2047(G_BASE - .012), lambda: total_2047(G_BASE + .012)),
             "Urbanisation 45%–57%": (lambda: total_2047(growth(0, 45)), lambda: total_2047(growth(0, 57))),
             "Population 1.57–1.70 bn": (lambda: total_2047(pop=1.57), lambda: total_2047(pop=1.70))}
    return {k: (lo() - total0, hi() - total0) for k, (lo, hi) in cases.items()}

if __name__ == "__main__":
    base = total_2047()
    print(f"Base 2047: {base:,.0f} TWh ({base/1824.1:.2f}x today)")
    print(f"Low/High : {total_2047(G_BASE-.012):,.0f} / {total_2047(G_BASE+.012):,.0f} TWh")
    print("\nTornado (delta TWh):")
    for k, (lo, hi) in tornado(base).items(): print(f"  {k:32s} {lo:+8.0f} {hi:+8.0f}")
    m, z = mix(base), zones(base)
    print("\n2047 mix (TWh):", {k: round(v) for k, v in m.items()})
    print("2047 zones (TWh):", {k: round(v) for k, v in z.items()})
    with open("outputs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["zone", "gen_2025_TWh", "gen_2047_TWh"])
        for k in ZONE_NOW: w.writerow([k, ZONE_NOW[k], round(z[k], 1)])
