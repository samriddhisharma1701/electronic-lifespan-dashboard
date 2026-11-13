import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import weibull_min

st.set_page_config(page_title="Reliability Studio", page_icon="📊", layout="wide")

# Consistent light theme and readable controls
st.markdown("""
<style>
:root { color-scheme: light; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    background: #f5f7fb !important; color: #17243a !important;
}
[data-testid="stHeader"] { background: rgba(245,247,251,.92) !important; }
[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] * { color: #26364d !important; }
.block-container { max-width: 1380px; padding-top: 2rem; padding-bottom: 3rem; }
h1,h2,h3 { color: #17243a !important; letter-spacing: -.02em; }
p, label, .stMarkdown { color: #52627a; }
div[data-testid="stMetric"] {
  background: white; border: 1px solid #e3e9f2; padding: 18px 20px;
  border-radius: 14px; box-shadow: 0 2px 8px rgba(25,45,75,.04);
}
div[data-testid="stMetric"] label { color: #61718a !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #17243a !important; }
.stButton button, .stDownloadButton button {
  border-radius: 9px; border: 1px solid #d7e0ed; font-weight: 600;
}
[data-testid="stFileUploader"] { background: white; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 12px; }
hr { border-color: #e3e9f2; }
.small-note { color:#718096; font-size:.88rem; }
.info-card {background:white;border:1px solid #e3e9f2;border-radius:14px;padding:18px 20px;margin-bottom:12px;}
.kicker {color:#55749d;font-size:.76rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def demo_data():
    rng = np.random.default_rng(12)
    params = {"Capacitor": (2.0, 10500), "Cooling Fan": (2.5, 15000),
              "LED Bulb": (3.0, 18000), "Resistor": (4.0, 32000)}
    rows = []
    for name, (beta, eta) in params.items():
        vals = weibull_min.rvs(beta, scale=eta, size=36, random_state=rng)
        for i, val in enumerate(vals, 1):
            rows.append({"component": name, "failure_time_hours": round(float(val), 1),
                         "unit_id": f"{name[:2].upper()}-{i:03d}"})
    return pd.DataFrame(rows)

def normalize_data(df):
    cols = {str(c).strip().lower(): c for c in df.columns}
    comp = next((cols[k] for k in ["component","component_name","part"] if k in cols), None)
    time = next((cols[k] for k in ["failure_time_hours","failure_time","time","hours","operating_hours"] if k in cols), None)
    if comp is None or time is None:
        raise ValueError("CSV must contain a component column and a failure-time column (hours).")
    out = df[[comp, time]].copy()
    out.columns = ["component", "failure_time_hours"]
    out["component"] = out["component"].astype(str).str.strip()
    out["failure_time_hours"] = pd.to_numeric(out["failure_time_hours"], errors="coerce")
    out = out.dropna()
    out = out[(out["component"] != "") & (out["failure_time_hours"] > 0)]
    if out.empty:
        raise ValueError("No valid positive failure times were found.")
    return out

def fit_weibull(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) < 3:
        return None
    beta, loc, eta = weibull_min.fit(vals, floc=0)
    return float(beta), float(eta)

def reliability(t, beta, eta):
    return np.exp(-((np.maximum(np.asarray(t), 0) / eta) ** beta))

if "data" not in st.session_state:
    st.session_state.data = demo_data()
if "source" not in st.session_state:
    st.session_state.source = "Simulated demonstration data"

with st.sidebar:
    st.markdown("## 📊 Reliability Studio")
    st.caption("Electronic component lifespan analysis")
    st.divider()
    page = st.radio("WORKSPACE", ["Overview", "Lifespan Predictor", "Reliability Analysis",
                                  "Data & Fitting", "Component Comparison", "Methodology"],
                    label_visibility="visible")
    st.divider()
    st.markdown("### DATA SOURCE")
    st.caption(st.session_state.source)
    uploaded = st.file_uploader("Upload failure-time CSV", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state.data = normalize_data(pd.read_csv(uploaded))
            st.session_state.source = f"Uploaded: {uploaded.name}"
            st.success("Data loaded.")
        except Exception as e:
            st.error(str(e))
    if st.button("Restore demo dataset", use_container_width=True):
        st.session_state.data = demo_data()
        st.session_state.source = "Simulated demonstration data"
        st.rerun()
    st.divider()
    st.caption("Model: 2-parameter Weibull distribution (location fixed at zero).")
    st.caption("All times are expressed in hours.")

df = st.session_state.data.copy()
components = sorted(df["component"].unique())
fits = {c: fit_weibull(df.loc[df.component == c, "failure_time_hours"]) for c in components}
valid_fits = {k:v for k,v in fits.items() if v is not None}

def header(eyebrow, title, subtitle):
    st.markdown(f'<div class="kicker">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.write(subtitle)

def plot_layout(fig, height=390):
    fig.update_layout(height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#42526a", family="Arial"), margin=dict(l=30,r=20,t=35,b=35),
        legend=dict(orientation="h", y=-.22), hovermode="x unified")
    fig.update_xaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

if page == "Overview":
    header("RELIABILITY WORKSPACE", "Electronic component lifespan",
           "Explore how long components operate before failure using Weibull reliability analysis.")
    st.info("This workspace uses a two-parameter Weibull model. The built-in dataset is simulated for demonstration; replace it with verified test observations for a real study.")
    a,b,c,d = st.columns(4)
    a.metric("Observations", f"{len(df):,}")
    b.metric("Components", len(components))
    c.metric("Median recorded time", f"{df.failure_time_hours.median():,.0f} h")
    d.metric("Successful model fits", f"{len(valid_fits)}/{len(components)}")
    left,right = st.columns([1.35,1])
    with left:
        st.subheader("Reliability curves")
        fig = go.Figure()
        for name,(beta,eta) in valid_fits.items():
            t = np.linspace(0, max(df.failure_time_hours)*1.05, 180)
            fig.add_trace(go.Scatter(x=t,y=reliability(t,beta,eta)*100,mode="lines",name=name,line=dict(width=3)))
        fig.update_layout(xaxis_title="Operating time (hours)", yaxis_title="Reliability (%)", yaxis_range=[0,100])
        plot_layout(fig,390)
    with right:
        st.subheader("Dataset snapshot")
        summary = df.groupby("component").failure_time_hours.agg(Records="count", Median="median", Mean="mean").reset_index()
        summary[["Median","Mean"]] = summary[["Median","Mean"]].round(0).astype(int)
        st.dataframe(summary, hide_index=True, use_container_width=True, height=300)
        st.caption("Statistics describe the supplied observations, not guaranteed future life.")
    st.subheader("How to use this website")
    cols = st.columns(3)
    for col, n, title, desc in zip(cols,["01","02","03"],
        ["Choose a component","Set operating conditions","Understand reliability"],
        ["Open the predictor and select a component.","Enter operating hours to estimate survival and failure probability.","Compare the curves and read the Weibull parameters."]):
        with col:
            st.markdown(f'<div class="info-card"><div class="kicker">{n}</div><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)

elif page == "Lifespan Predictor":
    header("INTERACTIVE TOOL", "Lifespan predictor",
           "Estimate the probability that a selected component will still be operating after a chosen time.")
    if not valid_fits:
        st.warning("At least three observations are needed for a component.")
    else:
        c1,c2 = st.columns([1,1])
        with c1:
            selected = st.selectbox("Select component", list(valid_fits))
            beta,eta = valid_fits[selected]
            hours = st.number_input("Operating time (hours)", min_value=0, value=int(eta), step=500)
        with c2:
            st.markdown("#### Model parameters")
            st.metric("Shape parameter (β)", f"{beta:.3f}")
            st.metric("Scale / characteristic life (η)", f"{eta:,.0f} h")
        r = float(reliability(hours,beta,eta))
        fail = 1-r
        x,y = st.columns(2)
        x.metric("Probability of survival", f"{r*100:.2f}%")
        y.metric("Probability of failure", f"{fail*100:.2f}%")
        st.progress(float(np.clip(r,0,1)), text=f"Estimated reliability at {hours:,} hours")
        st.markdown("#### What does this mean?")
        st.write(f"According to the fitted model, approximately **{r*100:.1f}%** of similar {selected.lower()} units would be expected to survive beyond **{hours:,} operating hours**. This is a model-based estimate, not a guarantee for an individual unit.")
        t = np.linspace(0,max(eta*2.2,hours*1.15,1),200)
        fig=go.Figure(go.Scatter(x=t,y=reliability(t,beta,eta)*100,mode="lines",name="Reliability",line=dict(width=3)))
        fig.add_vline(x=hours,line_dash="dash",line_color="#e07a35",annotation_text="Selected time")
        fig.update_layout(xaxis_title="Operating time (hours)",yaxis_title="Reliability (%)",yaxis_range=[0,100])
        plot_layout(fig,360)

elif page == "Reliability Analysis":
    header("MODEL OUTPUT", "Reliability analysis",
           "View reliability, cumulative failure probability and failure rate over operating time.")
    if valid_fits:
        selected=st.selectbox("Component",list(valid_fits))
        beta,eta=valid_fits[selected]
        t=np.linspace(1,max(eta*2.5,1),240)
        R=reliability(t,beta,eta)
        F=1-R
        hazard=(beta/eta)*(t/eta)**(beta-1)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=t,y=R*100,name="Reliability R(t)",line=dict(width=3)))
        fig.add_trace(go.Scatter(x=t,y=F*100,name="Failure probability F(t)",line=dict(width=3)))
        fig.update_layout(xaxis_title="Operating time (hours)",yaxis_title="Probability (%)",yaxis_range=[0,100])
        plot_layout(fig,380)
        fig2=go.Figure(go.Scatter(x=t,y=hazard,mode="lines",name="Hazard rate",line=dict(width=3)))
        fig2.update_layout(xaxis_title="Operating time (hours)",yaxis_title="Hazard rate (per hour)")
        plot_layout(fig2,300)
        st.markdown("**Interpretation:** Reliability is the chance of surviving beyond time t. Failure probability is 1 − R(t). The hazard rate describes the instantaneous failure tendency among units that have survived to time t.")
        st.write(f"For **{selected}**, β = {beta:.3f}. A β below 1 suggests decreasing failure rate; β near 1 suggests roughly constant failure rate; β above 1 suggests increasing failure rate.")

elif page == "Data & Fitting":
    header("DATA WORKSPACE", "Data and Weibull fitting",
           "Inspect the observations used to estimate the Weibull model.")
    st.caption(f"Current source: {st.session_state.source}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download current dataset",df.to_csv(index=False).encode("utf-8"),
                       "reliability_data.csv","text/csv")
    rows=[]
    for name,val in valid_fits.items():
        beta,eta=val
        rows.append({"Component":name,"Observations":int((df.component==name).sum()),
                     "Shape β":round(beta,4),"Scale η (hours)":round(eta,1),
                     "B10 life (hours)":round(eta*(-np.log(.9))**(1/beta),1)})
    st.subheader("Fitted Weibull parameters")
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("**Fitting method:** maximum likelihood estimation (MLE), with the location parameter fixed at zero.")
    st.warning("Each row is treated as an observed failure. Right-censored observations (units that have not failed by the end of testing) are not supported in this version.")

elif page == "Component Comparison":
    header("SIDE-BY-SIDE VIEW", "Component comparison",
           "Compare fitted lifetimes to identify which components have longer estimated operating lives.")
    if valid_fits:
        chosen=st.multiselect("Choose components",list(valid_fits),default=list(valid_fits))
        fig=go.Figure()
        for name in chosen:
            beta,eta=valid_fits[name]
            t=np.linspace(0,max(df.failure_time_hours)*1.05,180)
            fig.add_trace(go.Scatter(x=t,y=reliability(t,beta,eta)*100,mode="lines",name=name,line=dict(width=3)))
        fig.update_layout(xaxis_title="Operating time (hours)",yaxis_title="Reliability (%)",yaxis_range=[0,100])
        plot_layout(fig,420)
        rows=[]
        for name in chosen:
            beta,eta=valid_fits[name]
            rows.append({"Component":name,"Shape β":round(beta,3),"Characteristic life η (h)":round(eta),
                         "B10 life (h)":round(eta*(-np.log(.9))**(1/beta))})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.caption("B10 life is the estimated time by which 10% of the population is expected to have failed.")

else:
    header("PROJECT GUIDE", "Methodology",
           "A short explanation of the Weibull model and how this dashboard uses it.")
    st.markdown("### 1. Collect failure-time data")
    st.write("Record the operating time (in hours) at which each component fails. The demo data is simulated and is only for demonstrating the calculations.")
    st.markdown("### 2. Fit the Weibull distribution")
    st.latex(r"f(t) = \frac{\beta}{\eta}\left(\frac{t}{\eta}\right)^{\beta-1}e^{-(t/\eta)^\beta}")
    st.write("The shape parameter β controls the pattern of failures. The scale parameter η is the characteristic life, at which about 63.2% of the population has failed.")
    st.markdown("### 3. Calculate reliability")
    st.latex(r"R(t) = e^{-(t/\eta)^\beta}")
    st.write("Reliability R(t) is the probability that a component operates beyond time t. Failure probability is F(t) = 1 − R(t).")
    st.markdown("### 4. Interpret the shape parameter")
    st.write("- β < 1: decreasing failure rate (early-life failures)\n- β = 1: approximately constant failure rate\n- β > 1: increasing failure rate (wear-out behaviour)")
    st.warning("This is an educational analysis tool, not a certified engineering or warranty prediction system. Results depend on data quality and model assumptions.")
