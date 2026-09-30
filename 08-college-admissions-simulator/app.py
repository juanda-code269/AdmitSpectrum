import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Admissions Uncertainty Lab",page_icon="🎓",layout="wide")
st.title("Admissions Uncertainty Lab")
st.warning("This is a synthetic probability simulation—not a college chance calculator and not calibrated to any university.")


def inputs(prefix=""):
    c1,c2=st.columns(2)
    gpa=c1.slider("GPA (0–4)",0.0,4.0,3.5,.05,key=prefix+"gpa")
    rigor=c2.slider("Course rigor (simulated 1–5)",1,5,3,key=prefix+"rigor")
    test=c1.slider("Standardized score percentile",1,99,75,key=prefix+"test")
    activities=c2.slider("Activity strength (simulated 1–5)",1,5,3,key=prefix+"activities")
    essays=c1.slider("Essay strength (simulated 1–5)",1,5,3,key=prefix+"essays")
    context=c2.slider("Context/opportunity index (simulated 1–5)",1,5,3,key=prefix+"context")
    return np.array([gpa,rigor,test,activities,essays,context],dtype=float)


def simulate(x,n=10000,seed=1):
    rng=np.random.default_rng(seed)
    centered=np.array([(x[0]-3.2)/.55,(x[1]-3)/1.3,(x[2]-60)/25,(x[3]-3)/1.3,(x[4]-3)/1.3,(x[5]-3)/1.3])
    weights=np.array([.72,.42,.32,.24,.42,.20])
    applicant=centered@weights
    pool=rng.normal(0,.8,n); institutional=rng.normal(0,1.0,n); reader=rng.normal(0,.55,n)
    selectivity=rng.normal(1.15,.35,n)
    latent=applicant-pool+institutional+reader-selectivity
    return 1/(1+np.exp(-latent)), latent>0


tab1,tab2,tab3,tab4=st.tabs(["Scenario","Compare scenarios","Applicant overlap","Method & limits"])
with tab1:
    x=inputs("main_");draws,outcome=simulate(x,seed=42)
    q=np.quantile(draws,[.1,.25,.5,.75,.9]);a,b,c=st.columns(3)
    a.metric("Median simulated propensity",f"{q[2]:.0%}");b.metric("Middle 50% range",f"{q[1]:.0%}–{q[3]:.0%}");c.metric("Simulated admit frequency",f"{outcome.mean():.0%}")
    st.plotly_chart(px.histogram(draws,nbins=45,title="Distribution across uncertain pools, readers, and institutional scenarios",labels={"value":"Simulated propensity"}),width="stretch")
    st.caption("These values describe this invented model only. They are not real admissions probabilities.")
with tab2:
    c1,c2=st.columns(2)
    with c1:st.subheader("Scenario A");xa=inputs("a_")
    with c2:st.subheader("Scenario B");xb=inputs("b_")
    pa,_=simulate(xa,seed=8);pb,_=simulate(xb,seed=8)
    comp=pd.DataFrame({"A":pa,"B":pb}).melt(var_name="Scenario",value_name="Simulated propensity")
    st.plotly_chart(px.box(comp,x="Scenario",y="Simulated propensity",points=False,title="Same uncertain environments, different profiles"),width="stretch")
with tab3:
    rng=np.random.default_rng(10);n=5000
    population=pd.DataFrame({"GPA":np.clip(rng.normal(3.35,.42,n),0,4),"Essay":rng.integers(1,6,n),"Rigor":rng.integers(1,6,n)})
    latent=(population.GPA-3.2)*1.2+(population.Essay-3)*.35+(population.Rigor-3)*.3+rng.normal(0,1.4,n)-1
    population["Simulated outcome"]=np.where(latent>0,"Admitted","Not admitted")
    st.plotly_chart(px.scatter(population.sample(1200,random_state=1),x="GPA",y="Essay",color="Simulated outcome",opacity=.55,title="Overlapping profiles produce different outcomes"),width="stretch")
    st.write("Even in a synthetic world with known rules, unobserved factors create substantial overlap. Similar inputs do not imply identical outcomes.")
with tab4:
    labels=["GPA","Rigor","Test percentile","Activities","Essays","Context"]
    base=np.array([3.5,3,75,3,3,3],float);base_p=simulate(base,seed=3)[0].mean();rows=[]
    steps=[.25,1,10,1,1,1]
    for i,label in enumerate(labels):
        changed=base.copy();changed[i]+=steps[i];rows.append({"Input":label,"Illustrative change in model output":simulate(changed,seed=3)[0].mean()-base_p})
    st.plotly_chart(px.bar(pd.DataFrame(rows),x="Input",y="Illustrative change in model output",title="Local sensitivity of the invented formula"),width="stretch")
    st.markdown("""### What the simulation omits
Real decisions depend on applicant-pool strength, institutional priorities, recommendations, essays, major capacity, geography, resources and opportunity, and many contextual factors that cannot be represented faithfully here. The coefficients are deliberately synthetic and imply **no university-specific probability**. Association in observational admissions data would not establish that changing one input causes admission. Standardized tests are included only as a hypothetical input and may be irrelevant at test-optional institutions.""")

