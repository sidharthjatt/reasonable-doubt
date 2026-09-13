"""Does the margin detect out-of-distribution input? Measured, and the answer is no.

Twenty hand-written non-contract paragraphs sent to a running service. These are
PROBES, not a sample from any distribution: the rate below characterises these
twenty inputs and nothing wider. Registered as PREREGISTRATION 3bs.

    python scripts/ood_probe.py [base_url]
"""
import collections
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
PROBES = {
 "credit-risk model card": "This model estimates credit risk for retail borrowers. Inputs include income, utilisation and delinquency history. Performance is monitored quarterly and the model is recalibrated when the population stability index exceeds the agreed tolerance.",
 "credit-risk methodology": "Expected loss is decomposed into probability of default, loss given default and exposure at default. Each component is estimated separately and the resulting provision is reviewed by the model risk function.",
 "credit-risk governance": "The methodology is documented, independently validated and subject to annual review. Material changes require approval before deployment into the production scoring environment.",
 "software architecture": "The service is deployed as a stateless container behind a load balancer. Requests are authenticated at the edge and routed to the nearest healthy replica, with retries handled by the client library.",
 "ML abstract": "We propose a transformer architecture that attends over spectral features, achieving a 3.2 point improvement on the benchmark while reducing inference latency by forty percent.",
 "data policy prose": "Access to production data is restricted to named individuals and reviewed on a quarterly basis. All access is logged and retained for eighteen months.",
 "weather report": "Scattered thunderstorms are expected across the region this afternoon, with rainfall totals of 20 to 40 millimetres and gusts reaching 60 kilometres per hour.",
 "medical note": "The patient reports intermittent chest discomfort on exertion, relieved by rest. ECG shows no acute ST changes. Troponin is within normal limits.",
 "restaurant review": "The pasta was overcooked and the sauce lacked seasoning, though the service was attentive and the room pleasant enough for a weekday lunch.",
 "cake recipe": "Please preheat the oven to 180 degrees Celsius and bake the cake for 35 minutes until golden brown.",
 "football report": "United equalised in the 78th minute through a deflected free kick, and the visitors held on for a point despite finishing with ten men.",
 "poem": "I wandered lonely as a cloud that floats on high o'er vales and hills, when all at once I saw a crowd, a host of golden daffodils.",
 "news lede": "The central bank raised its benchmark rate by twenty-five basis points on Thursday, citing persistent inflation in services and a tight labour market.",
 "python docstring": "Returns a list of integers parsed from the input string, raising ValueError if any token cannot be converted.",
 "travel blurb": "The old town is best explored on foot in the early morning, before the cruise passengers arrive and the narrow lanes fill up.",
 "job ad": "We are looking for a backend engineer with strong Python and distributed systems experience to join a small team building data infrastructure.",
 "cooking method": "Dice the onions finely and sweat them in olive oil over a low heat for about ten minutes until translucent but not coloured.",
 "physics": "The observed redshift implies a recession velocity proportional to distance, consistent with uniform expansion at the measured Hubble parameter.",
 "history": "The treaty was signed in the spring of that year, ending three decades of intermittent conflict and redrawing the frontier along the river.",
 "product copy": "Lightweight, water resistant and packs down to the size of a paperback. Ideal for commuting or weekend trips.",
}
rows = []
for name, text in PROBES.items():
    req = urllib.request.Request(BASE + "/classify",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req).read())
    rows.append((name, d["label"], d["top_3"][0]["score"], d["margin"], d["needs_review"]))
rows.sort(key=lambda r: -r[3])
print(f"{'input':<24}{'label':<22}{'top1':>7}{'margin':>9}  flag")
for n, l, s, m, f in rows:
    print(f"{n:<24}{l:<22}{s*100:>6.1f}%{m:>9.4f}  {'' if f else 'NOT FLAGGED'}")
un = [r for r in rows if not r[4]]
print(f"\n{len(rows)-len(un)}/{len(rows)} flagged.  {len(un)}/{len(rows)} "
      f"({100*len(un)/len(rows):.0f}%) answered CONFIDENTLY with no flag.")
print("labels the unflagged ones landed on:",
      dict(collections.Counter(r[1] for r in un)))
print("all labels seen:", dict(collections.Counter(r[1] for r in rows)))
