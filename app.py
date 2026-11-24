import pandas as pd
import streamlit as st

from algorithms import (
    bayesian_occupancy_update,
    hill_climbing_schedule,
    mdp_appliance_decision,
)

st.set_page_config(page_title="Mini Smart Home Energy Saver", layout="wide")
st.title("Mini Smart Home Energy Saver")
st.caption("AI Fundamentals Project: Hill Climbing + Bayesian Inference + MDP")

st.sidebar.header("Simulation Parameters")

prior_occ = st.sidebar.slider("Prior P(Occupied)", 0.05, 0.95, 0.30, 0.05)
p_alert_occ = st.sidebar.slider("P(Alert | Occupied)", 0.50, 0.99, 0.85, 0.01)
p_alert_unocc = st.sidebar.slider("P(Alert | Unoccupied)", 0.01, 0.50, 0.15, 0.01)
sensor_alert = st.sidebar.selectbox("Sensor Reading", ["Alert (Motion)", "No Alert"]) == "Alert (Motion)"

st.sidebar.subheader("MDP Rewards")
r_keep_occ = st.sidebar.number_input("Keep On | Occupied", value=10.0)
r_keep_unocc = st.sidebar.number_input("Keep On | Unoccupied", value=-10.0)
r_off_occ = st.sidebar.number_input("Turn Off | Occupied", value=-5.0)
r_off_unocc = st.sidebar.number_input("Turn Off | Unoccupied", value=5.0)

schedules = {
    1: {"name": "Slot 1 (08:00-12:00)", "cost": 15.5},
    2: {"name": "Slot 2 (10:00-14:00)", "cost": 11.2},
    3: {"name": "Slot 3 (12:00-16:00)", "cost": 8.5},
    4: {"name": "Slot 4 (14:00-18:00)", "cost": 12.0},
    5: {"name": "Slot 5 (16:00-20:00)", "cost": 14.1},
}

best_schedule, hc_steps = hill_climbing_schedule(schedules, start_id=1)
bayes_res = bayesian_occupancy_update(prior_occ, p_alert_occ, p_alert_unocc, sensor_alert)
rewards = {
    "Keep On": {"Occupied": r_keep_occ, "Unoccupied": r_keep_unocc},
    "Turn Off": {"Occupied": r_off_occ, "Unoccupied": r_off_unocc},
}
best_action, eu_map = mdp_appliance_decision(bayes_res["posterior_occupied"], rewards)

c1, c2, c3 = st.columns(3)
c1.metric("Best Schedule (Hill Climbing)", best_schedule["name"], f"Cost: {best_schedule['cost']}")
c2.metric("Posterior P(Occupied)", f"{bayes_res['posterior_occupied']:.2f}", f"Prior: {prior_occ:.2f}")
c3.metric("Recommended Action (MDP)", best_action, f"EU: {eu_map[best_action]:.2f}")

st.divider()

st.subheader("1. Hill Climbing — Schedule Cost Optimization")
df_sched = pd.DataFrame(
    [{"Slot": v["name"], "Energy Cost": v["cost"]} for v in schedules.values()]
)
st.bar_chart(df_sched.set_index("Slot"))
st.dataframe(pd.DataFrame(hc_steps), use_container_width=True)
st.info("Explanation: Hill Climbing moves along neighboring schedule slots to find the lowest energy cost.")

st.subheader("2. Bayesian Inference — Occupancy Probability")
st.write(
    f"Prior: **{bayes_res['prior_occupied']:.2f}** | "
    f"Evidence P(Obs): **{bayes_res['p_evidence']:.3f}** | "
    f"Posterior P(Occupied): **{bayes_res['posterior_occupied']:.2f}**"
)
st.info("Explanation: Bayesian Inference updates occupancy probability using prior belief and the current sensor reading.")

st.subheader("3. MDP — Expected Utility Decision")
df_eu = pd.DataFrame(
    [{"Action": act, "Expected Utility": round(val, 2)} for act, val in eu_map.items()]
)
st.dataframe(df_eu, use_container_width=True)
st.info("Explanation: MDP picks the action with the higher expected utility given the updated occupancy probability.")
