"""Insurance Claim Fraud Investigation Dashboard — UX/Analytics refresh."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent if (HERE.parent / "src").exists() else HERE
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.predict import score_claims, REQUIRED_INFERENCE_COLUMNS  # noqa: E402
from src.risk_engine import build_investigation_queue, summarize_queue  # noqa: E402

st.set_page_config(page_title="Claim Fraud Intelligence", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1500px}
[data-testid="stMetric"]{background:rgba(120,130,150,.09);border:1px solid rgba(120,130,150,.18);padding:14px 16px;border-radius:13px}
[data-testid="stMetricLabel"]{font-size:.9rem}[data-testid="stMetricValue"]{font-weight:700}
section[data-testid="stSidebar"]{border-right:1px solid rgba(120,130,150,.15)}
.hero{padding:1.25rem 1.5rem;border:1px solid rgba(120,130,150,.2);border-radius:18px;margin-bottom:1rem;background:linear-gradient(115deg,rgba(40,110,210,.14),rgba(90,110,160,.04))}
.hero h1{margin:0 0 .35rem 0;font-size:2rem}.hero p{margin:0;opacity:.8}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🛡️ Insurance Claim Fraud Intelligence</h1><p>Risk-based investigation prioritization · Human-in-the-loop decision support · Not an automated rejection system</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Analysis Controls")
    uploaded = st.file_uploader("Upload claims CSV", type=["csv"])
    capacity = st.number_input("Investigation capacity", min_value=1, max_value=100000, value=200, step=10, help="Maximum claims to prioritize for this review period.")
    threshold = st.slider("Predicted-fraud label threshold", 0.0, 1.0, 0.50, 0.01, help="Changes the Predicted_Fraud label only. The queue is ranked by risk score and capacity.")
    st.divider()
    st.caption("Model: Random Forest risk-scoring pipeline")
    st.caption("Historical educational data; not validated for current operational use.")

if uploaded is None:
    st.info("Upload a claims CSV to score claims and build a prioritized investigation queue.")
    with st.expander("Required input fields", expanded=True):
        st.caption("FraudFound_P, if present, is excluded from scoring and used only for retrospective analysis.")
        st.code("\n".join(REQUIRED_INFERENCE_COLUMNS), language="text")
    st.markdown("**Suggested workflow**: Upload → Review overview → Explore risk → Investigate queue → Claim review → Export")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}"); st.stop()
if raw.empty:
    st.error("The uploaded file contains no rows."); st.stop()
missing = [col for col in REQUIRED_INFERENCE_COLUMNS if col not in raw.columns]
if missing:
    st.error("Missing required fields:"); st.write(missing); st.stop()

# Keep historical labels separate from the scoring inputs.
has_target = "FraudFound_P" in raw.columns
actual_target = raw["FraudFound_P"].copy() if has_target else None
with st.spinner("Scoring claims and ranking investigation candidates…"):
    try:
        scored = score_claims(raw, threshold=float(threshold))
        queue = build_investigation_queue(scored, capacity=int(capacity))
        summary = summarize_queue(queue)
    except Exception as exc:
        st.error(f"Scoring/queue processing failed: {exc}"); st.exception(exc); st.stop()
selected = queue.loc[queue["Investigation_Selected"]].copy()
N = int(summary["total_claims"]); Q = int(summary["selected_for_investigation"])
outside_queue_count = N - Q
coverage = Q/N*100 if N else 0.0
utilization = (Q / int(capacity) * 100) if int(capacity) > 0 else 0.0
mean_risk = summary.get("mean_risk_score_selected")
boundary = summary.get("min_risk_score_selected")
max_selected_risk = summary.get("max_risk_score_selected")
high30 = int(scored["Fraud_Risk_Score"].ge(.30).sum())
high50 = int(scored["Fraud_Risk_Score"].ge(.50).sum())

st.subheader("Executive Overview")
a,b,c,d=st.columns(4)
a.metric("Claims processed",f"{N:,}")
b.metric("Investigation queue",f"{Q:,}")
c.metric("Capacity utilization",f"{utilization:.1f}%",help="Queue size divided by the configured capacity. 100% means all available slots are filled.")
d.metric("Investigation coverage",f"{coverage:.1f}%",help="Selected claims as a percentage of all claims processed.")
e,f,g,h=st.columns(4)
e.metric("Average queue risk",f"{mean_risk:.1%}" if mean_risk is not None else "N/A")
f.metric("High-risk claims ≥30%",f"{high30:,}")
g.metric("Very-high-risk claims ≥50%",f"{high50:,}")
h.metric("Queue boundary score",f"{boundary:.1%}" if boundary is not None else "N/A",help="Score of the lowest-risk claim selected into the queue.")
if Q:
    st.info(f"**Operational Summary**\n\nProcessed **{N:,} claims** and prioritized **{Q:,}** for investigation, utilizing **{utilization:.1f}%** of configured capacity. The queue's average predicted risk is **{mean_risk:.1%}** and its selection boundary is **{boundary:.1%}**. Scores are model estimates—not confirmed fraud.")
else:
    st.warning("No claims were selected. Check the capacity setting and input data.")

# Tabs reorganize analytics, exploration, review and reporting into one experience.
tabs = st.tabs(["📊 Overview", "📈 Risk Analytics", "🧩 Fraud Patterns", "🔎 Investigation Queue", "🧾 Claim Review", "⬇️ Export"])

# Overview
with tabs[0]:
    st.subheader("Risk profile at a glance")
    l,r=st.columns(2)
    band_order=["Low","Moderate","High","Very High"]
    bands=scored["Risk_Band"].value_counts().reindex(band_order,fill_value=0)
    with l:
        st.markdown("**Claims by risk band**")
        st.bar_chart(bands, horizontal=True)
        st.caption("Risk bands: Low <10%; Moderate 10–30%; High 30–50%; Very High ≥50%.")
    with r:
        st.markdown("**Queue priority composition**")
        if "Priority" in queue:
            priority_order=["Critical","High","Moderate","Low"]
            pcounts=selected["Priority"].value_counts().reindex(priority_order,fill_value=0)
            st.bar_chart(pcounts, horizontal=True)
        st.markdown("**Queue score range**")
        st.write(f"Highest selected score: **{max_selected_risk:.1%}**" if max_selected_risk is not None else "Highest selected score: N/A")
        st.write(f"Lowest selected score: **{boundary:.1%}**" if boundary is not None else "Lowest selected score: N/A")
        st.write(f"Claims not selected: **{N-Q:,}**")
    st.divider()
    st.markdown("**Decision-support reminder**")
    st.warning("Use risk scores to prioritize human review. A score is not proof of fraud and must not be the sole reason for rejecting, delaying, or penalizing a claim.")

# Risk analytics
with tabs[1]:
    st.subheader("Risk distribution analytics")
    ra,rb,rc=st.columns(3)
    ra.metric("Median risk score",f"{scored['Fraud_Risk_Score'].median():.1%}")
    rb.metric("90th percentile",f"{scored['Fraud_Risk_Score'].quantile(.90):.1%}")
    rc.metric("Claims ≥ selected threshold",f"{int(scored['Fraud_Risk_Score'].ge(threshold).sum()):,}")
    st.markdown("**Risk-score distribution (percentage bins)**")
    bins=[0,.05,.10,.20,.30,.40,.50,.60,.70,.80,.90,1.000001]
    hist=pd.cut(scored["Fraud_Risk_Score"],bins=bins,include_lowest=True).value_counts().sort_index()
    hist.index=[f"{int(x.left*100)}–{int(x.right*100)}%" for x in hist.index]
    st.bar_chart(hist, horizontal=False)
    st.markdown("**Risk band table**")
    band_table=pd.DataFrame({"Risk Band":band_order,"Claims":bands.values,"Share of claims":(bands.values/N*100).round(2) if N else 0})
    st.dataframe(band_table,use_container_width=True,hide_index=True)
    st.markdown("**Capacity vs risk cut-off**")
    st.write(f"Threshold **{threshold:.0%}** labels **{int(scored['Fraud_Risk_Score'].ge(threshold).sum()):,}** claims as predicted fraud; the capacity-based queue still selects the top **{Q:,}** claims regardless of this threshold.")

# Pattern exploration (actual labels only if provided; otherwise score distributions)
with tabs[2]:
    st.subheader("Fraud / risk pattern exploration")
    st.caption("Explore group-level patterns. These are associations, not causal explanations or investigation rules.")
    group_candidates=["Fault","BasePolicy","PolicyType","VehicleCategory","Make","VehiclePrice","AgeOfVehicle","PastNumberOfClaims","AgeOfPolicyHolder","PoliceReportFiled","WitnessPresent","AccidentArea","AgentType","Sex","MaritalStatus"]
    available=[c for c in group_candidates if c in scored.columns]
    if not available:
        st.info("No suitable categorical fields were found.")
    else:
        chosen_group=st.selectbox("Break down claims by",available,index=0)
        pattern=scored.copy()
        pattern["Actual Fraud"] = actual_target.values if actual_target is not None else None
        agg=pattern.groupby(chosen_group,dropna=False).agg(Claims=("Fraud_Risk_Score","size"),MeanRisk=("Fraud_Risk_Score","mean"),HighRiskClaims=("Fraud_Risk_Score",lambda s:int(s.ge(.30).sum())))
        agg["ClaimSharePct"]=agg["Claims"]/N*100
        agg=agg.sort_values("MeanRisk",ascending=False)
        if actual_target is not None:
            # Convert the target safely for retrospective group rates.
            target_num=pd.to_numeric(actual_target,errors="coerce")
            pattern["Actual Fraud"] = target_num
            actual_rates=pattern.groupby(chosen_group,dropna=False)["Actual Fraud"].mean()
            agg["HistoricalFraudRate"]=actual_rates
        show=agg.reset_index()
        st.markdown("**Group summary**")
        st.dataframe(show,use_container_width=True,hide_index=True)
        st.markdown("**Average predicted risk by group**")
        st.bar_chart(agg["MeanRisk"].sort_values(ascending=False),horizontal=True)
        if actual_target is not None:
            st.markdown("**Historical fraud rate by group (target provided)**")
            st.bar_chart(agg["HistoricalFraudRate"].sort_values(ascending=False),horizontal=True)
            st.caption("Historical fraud rate is calculated from FraudFound_P in the uploaded file; it was not used as a model input.")
        else:
            st.info("For actual historical fraud-rate comparisons, upload a file that includes FraudFound_P. It will be excluded from model scoring.")

# Investigation queue
with tabs[3]:
    st.subheader("Investigation worklist")
    st.caption("Ranked queue is capacity constrained. Filters below change the displayed view—not the underlying selected queue.")
    q1,q2,q3=st.columns([1,1,1.4])
    with q1:
        priority_options=sorted(selected["Priority"].dropna().astype(str).unique().tolist()) if "Priority" in selected else []
        priorities=st.multiselect("Priority",priority_options,default=priority_options)
    with q2:
        risk_options=band_order
        risk_filter=st.multiselect("Risk band",risk_options,default=risk_options)
    with q3:
        search_text=st.text_input("Search claim / policy ID",placeholder="Type an ID or policy number")
    filtered=selected.copy()
    if "Priority" in filtered and priorities:
        filtered=filtered[filtered["Priority"].astype(str).isin(priorities)]
    if "Risk_Band" in filtered and risk_filter:
        filtered=filtered[filtered["Risk_Band"].astype(str).isin(risk_filter)]
    if search_text.strip():
        searchable=[c for c in ["PolicyNumber","Scoring_Row_ID"] if c in filtered.columns]
        if searchable:
            mask=pd.Series(False,index=filtered.index)
            for col in searchable: mask |= filtered[col].astype(str).str.contains(search_text.strip(),case=False,na=False)
            filtered=filtered[mask]
    st.write(f"Showing **{len(filtered):,}** of **{Q:,}** selected claims")
    preferred=["Queue_Position","Risk_Rank","PolicyNumber","Scoring_Row_ID","Fraud_Risk_Percent","Risk_Band","Priority","Investigation_Status","Make","VehicleCategory","PolicyType","Age","Fault"]
    display=[c for c in preferred if c in filtered.columns]
    st.dataframe(filtered[display],use_container_width=True,hide_index=True,height=460)
    st.download_button("Download filtered worklist",filtered.to_csv(index=False).encode("utf-8"),"filtered_investigation_worklist.csv","text/csv")

# Individual claim review
with tabs[4]:
    st.subheader("Claim review workspace")
    id_col="PolicyNumber" if "PolicyNumber" in queue.columns else "Scoring_Row_ID"
    choices=queue[id_col].astype(str).tolist()
    choice=st.selectbox("Choose a claim",choices,key="claim_review_select")
    claim=queue.loc[queue[id_col].astype(str)==choice].iloc[0]
    x1,x2,x3,x4=st.columns(4)
    x1.metric("Predicted risk",f"{claim['Fraud_Risk_Score']:.1%}")
    x2.metric("Risk band",str(claim.get("Risk_Band","N/A")))
    x3.metric("Priority",str(claim.get("Priority","N/A")))
    x4.metric("Queue position",str(claim.get("Queue_Position","Outside selected queue")))
    if bool(claim.get("Investigation_Selected",False)):
        st.success("This claim is included in the current investigation queue.")
    else:
        st.info("This claim is outside the current capacity-based investigation queue.")
    with st.expander("Claim attributes",expanded=True):
        details=claim.drop(labels=["Fraud_Risk_Score","Fraud_Risk_Percent","Predicted_Fraud"],errors="ignore")
        st.dataframe(details.to_frame("Value"),use_container_width=True)
    st.warning("Investigators must validate evidence and follow established procedures. The score must not be the sole basis for a claim decision.")

# Exports
with tabs[5]:
    st.subheader("Export & reporting")
    st.write("Download review-ready files for analysis or investigator handoff.")
    exp1,exp2=st.columns(2)
    with exp1:
        st.markdown("**Investigation queue**")
        st.caption(f"{Q:,} selected claims, ordered by risk score.")
        st.download_button("⬇️ Download selected queue",selected.to_csv(index=False).encode("utf-8"),"investigation_queue.csv","text/csv",use_container_width=True)
    with exp2:
        st.markdown("**All claims ranked**")
        st.caption(f"Complete ranked file containing {N:,} claims.")
        st.download_button("⬇️ Download all ranked claims",queue.to_csv(index=False).encode("utf-8"),"all_ranked_claims.csv","text/csv",use_container_width=True)
    report=pd.DataFrame([{
        "Claims_Processed":N,"Configured_Capacity":int(capacity),"Queue_Selected":Q,
        "Capacity_Utilization_Pct":round(utilization,2),"Investigation_Coverage_Pct":round(coverage,2),
        "Average_Queue_Risk":mean_risk,"Queue_Boundary_Score":boundary,
        "High_Risk_30_Plus":high30,"Very_High_Risk_50_Plus":high50,
        "Outside_Queue":outside_queue_count,"Label_Threshold":float(threshold)
    }])
    st.download_button("⬇️ Download KPI summary CSV",report.to_csv(index=False).encode("utf-8"),"dashboard_kpi_summary.csv","text/csv",use_container_width=True)
    st.caption("Exports contain model risk estimates and operational queue fields. Handle claim data according to privacy and access-control requirements.")

st.divider()
st.caption("Portfolio prototype using a historical educational dataset. Before deployment, assess model validation, calibration, fairness, drift, privacy, explainability, investigator outcomes, and financial impact.")
