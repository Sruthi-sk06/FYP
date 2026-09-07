"""
MOVENTRA - Module 3: Progress Dashboard
-----------------------------------------
This Streamlit app lets a therapist/caregiver view a stroke patient's
rehabilitation progress across sessions of the three Module 2 games:
Balloon Reach, Follow the Path, and Object Sorting.

IMPORTANT: This dashboard is a software-based progress monitoring tool
only. It does NOT provide medical diagnosis or clinical decisions.

Run with:
    streamlit run "module 3/dashboard.py"
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# 1. BASIC SETUP
# ---------------------------------------------------------------------------

# Use a path relative to this script's own folder, so the app works no
# matter where the project is placed on the computer (no hardcoded paths).
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(THIS_FOLDER, "patient_progress.csv")

# The exact columns our CSV file uses. Keeping this as a list makes it easy
# to create a correctly-shaped empty file if one doesn't exist yet.
CSV_COLUMNS = [
    "patient_id", "patient_name", "session_number", "session_date",
    "balloon_score", "balloon_accuracy", "balloon_attempts",
    "balloon_successful_targets", "balloon_time_sec",
    "path_score", "path_accuracy", "path_checkpoints_completed",
    "path_time_sec", "path_distance_moved",
    "sorting_score", "sorting_accuracy", "sorting_correct", "sorting_wrong",
    "sorting_targets_completed", "sorting_time_sec", "sorting_distance_moved",
    "is_sample_data",
]

st.set_page_config(
    page_title="MOVENTRA - Progress Dashboard",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------------------------
# 2. DATA HANDLING FUNCTIONS
# ---------------------------------------------------------------------------

def ensure_csv_exists():
    """Create patient_progress.csv with just headers if it doesn't exist yet.
    This means the app never crashes on a fresh computer / fresh clone."""
    if not os.path.exists(CSV_PATH):
        empty_df = pd.DataFrame(columns=CSV_COLUMNS)
        empty_df.to_csv(CSV_PATH, index=False)


def load_data():
    """Load session data from the CSV file safely.
    Returns an empty (but correctly shaped) DataFrame if the file is
    missing or empty, so the rest of the app never has to worry about it."""
    ensure_csv_exists()
    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    if df.empty:
        return df

    # Make sure numeric columns are actually numeric (protects graphs/maths
    # from crashing if a value was typed oddly).
    numeric_cols = [c for c in CSV_COLUMNS if c not in
                    ("patient_id", "patient_name", "session_date", "is_sample_data")]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_new_session(row_dict):
    """Append one new session (as a dict) to the CSV file."""
    df = load_data()
    new_row = pd.DataFrame([row_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)


def calculate_overall(balloon_score, path_score, sorting_score,
                       balloon_acc, path_acc, sorting_acc):
    """Simple, transparent overall metrics: plain averages of the three
    games. No hidden weighting or clinical scoring."""
    overall_score = round((balloon_score + path_score + sorting_score) / 3, 1)
    overall_accuracy = round((balloon_acc + path_acc + sorting_acc) / 3, 1)
    return overall_score, overall_accuracy


# ---------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------

st.sidebar.title("MOVENTRA")
st.sidebar.caption("Module 3 - Progress Dashboard")
page = st.sidebar.radio("Navigation", ["Dashboard", "Session History", "Add Session"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "This dashboard shows software-based progress tracking only. "
    "It is not a medical diagnostic tool."
)

# Load data once, used by every page.
data = load_data()

# ---------------------------------------------------------------------------
# 4. HEADER (shown on every page)
# ---------------------------------------------------------------------------

st.title("MOVENTRA")
st.subheader("Stroke Rehabilitation Progress Dashboard")

if data.empty:
    st.warning(
        "No session data found yet. Go to 'Add Session' to add the first "
        "session, or check that patient_progress.csv has data."
    )
    st.stop()

if data["is_sample_data"].astype(str).str.upper().eq("YES").any():
    st.info("Some records shown below are DEMONSTRATION SAMPLE DATA for Review 2.")


# ---------------------------------------------------------------------------
# 5. PATIENT SELECTION (used by Dashboard and Session History pages)
# ---------------------------------------------------------------------------

def patient_selector(df):
    """Shows a dropdown of patient IDs + names, returns the selected
    patient's rows (sorted by session number) and their ID."""
    patient_options = (
        df[["patient_id", "patient_name"]]
        .drop_duplicates()
        .sort_values("patient_id")
    )
    labels = [f"{row.patient_id} - {row.patient_name}" for row in patient_options.itertuples()]
    choice = st.selectbox("Select Patient", labels)
    selected_id = choice.split(" - ")[0]

    patient_rows = df[df["patient_id"] == selected_id].sort_values("session_number")
    return patient_rows, selected_id


# ---------------------------------------------------------------------------
# 6. PAGE: DASHBOARD
# ---------------------------------------------------------------------------

if page == "Dashboard":
    patient_rows, patient_id = patient_selector(data)
    latest = patient_rows.iloc[-1]  # most recent session for this patient

    st.markdown("### Patient Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Patient ID", latest["patient_id"])
    col2.metric("Patient Name", latest["patient_name"])
    col3.metric("Last Session Date", str(latest["session_date"]))

    # --- Summary cards (using the latest session's overall figures) -------
    overall_score, overall_accuracy = calculate_overall(
        latest["balloon_score"], latest["path_score"], latest["sorting_score"],
        latest["balloon_accuracy"], latest["path_accuracy"], latest["sorting_accuracy"],
    )
    total_time_sec = (
        latest["balloon_time_sec"] + latest["path_time_sec"] + latest["sorting_time_sec"]
    )

    st.markdown("### Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Score", f"{overall_score}")
    c2.metric("Overall Accuracy", f"{overall_accuracy}%")
    c3.metric("Total Sessions", int(patient_rows["session_number"].max()))
    c4.metric("Last Session Exercise Time", f"{int(total_time_sec)} sec")

    # --- Game performance cards/tables (latest session) -------------------
    st.markdown("### Game Performance (Latest Session)")
    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("#### 🎈 Balloon Reach")
        st.table(pd.DataFrame({
            "Metric": ["Score", "Accuracy", "Attempts", "Successful Targets", "Time (sec)"],
            "Value": [
                latest["balloon_score"], f"{latest['balloon_accuracy']}%",
                int(latest["balloon_attempts"]), int(latest["balloon_successful_targets"]),
                int(latest["balloon_time_sec"]),
            ],
        }).set_index("Metric"))

    with g2:
        st.markdown("#### 🛤️ Follow the Path")
        st.table(pd.DataFrame({
            "Metric": ["Score", "Path Accuracy", "Checkpoints Completed", "Time (sec)", "Distance Moved"],
            "Value": [
                latest["path_score"], f"{latest['path_accuracy']}%",
                int(latest["path_checkpoints_completed"]), int(latest["path_time_sec"]),
                latest["path_distance_moved"],
            ],
        }).set_index("Metric"))

    with g3:
        st.markdown("#### 🧩 Object Sorting")
        st.table(pd.DataFrame({
            "Metric": ["Score", "Accuracy", "Correct", "Wrong", "Targets Completed", "Time (sec)", "Distance Moved"],
            "Value": [
                latest["sorting_score"], f"{latest['sorting_accuracy']}%",
                int(latest["sorting_correct"]), int(latest["sorting_wrong"]),
                int(latest["sorting_targets_completed"]), int(latest["sorting_time_sec"]),
                latest["sorting_distance_moved"],
            ],
        }).set_index("Metric"))

    # --- Progress graphs: score & accuracy across sessions ----------------
    st.markdown("### Progress Across Sessions")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(patient_rows["session_number"], patient_rows["balloon_score"], marker="o", label="Balloon Reach")
    ax1.plot(patient_rows["session_number"], patient_rows["path_score"], marker="o", label="Follow the Path")
    ax1.plot(patient_rows["session_number"], patient_rows["sorting_score"], marker="o", label="Object Sorting")
    ax1.set_xlabel("Session Number")
    ax1.set_ylabel("Score")
    ax1.set_title("Score Progress by Game")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(patient_rows["session_number"], patient_rows["balloon_accuracy"], marker="o", label="Balloon Reach")
    ax2.plot(patient_rows["session_number"], patient_rows["path_accuracy"], marker="o", label="Follow the Path")
    ax2.plot(patient_rows["session_number"], patient_rows["sorting_accuracy"], marker="o", label="Object Sorting")
    ax2.set_xlabel("Session Number")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Progress by Game")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)


# ---------------------------------------------------------------------------
# 7. PAGE: SESSION HISTORY
# ---------------------------------------------------------------------------

elif page == "Session History":
    patient_rows, patient_id = patient_selector(data)

    st.markdown("### Session History")

    # Build the simple summary table requested:
    # Session | Date | Balloon Reach | Follow the Path | Object Sorting | Overall Score | Overall Accuracy
    history_rows = []
    for _, r in patient_rows.iterrows():
        overall_score, overall_accuracy = calculate_overall(
            r["balloon_score"], r["path_score"], r["sorting_score"],
            r["balloon_accuracy"], r["path_accuracy"], r["sorting_accuracy"],
        )
        history_rows.append({
            "Session": int(r["session_number"]),
            "Date": r["session_date"],
            "Balloon Reach": r["balloon_score"],
            "Follow the Path": r["path_score"],
            "Object Sorting": r["sorting_score"],
            "Overall Score": overall_score,
            "Overall Accuracy (%)": overall_accuracy,
        })

    history_df = pd.DataFrame(history_rows)
    st.dataframe(history_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 8. PAGE: ADD SESSION
# ---------------------------------------------------------------------------

elif page == "Add Session":
    st.markdown("### Add New Session")
    st.caption("Enter the results for one rehabilitation session. This will be saved to patient_progress.csv.")

    with st.form("add_session_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID", placeholder="e.g. P003")
            patient_name = st.text_input("Patient Name", placeholder="e.g. Suresh Menon")
        with col2:
            session_date = st.date_input("Date")

        st.markdown("#### 🎈 Balloon Reach")
        b1, b2 = st.columns(2)
        balloon_score = b1.number_input("Balloon Reach Score", min_value=0.0, step=1.0)
        balloon_accuracy = b2.number_input("Balloon Reach Accuracy (%)", min_value=0.0, max_value=100.0, step=1.0)

        st.markdown("#### 🛤️ Follow the Path")
        p1, p2 = st.columns(2)
        path_score = p1.number_input("Follow the Path Score", min_value=0.0, step=1.0)
        path_accuracy = p2.number_input("Follow the Path Accuracy (%)", min_value=0.0, max_value=100.0, step=1.0)

        st.markdown("#### 🧩 Object Sorting")
        s1, s2 = st.columns(2)
        sorting_score = s1.number_input("Object Sorting Score", min_value=0.0, step=1.0)
        sorting_accuracy = s2.number_input("Object Sorting Accuracy (%)", min_value=0.0, max_value=100.0, step=1.0)

        submitted = st.form_submit_button("Save Session")

    if submitted:
        if not patient_id or not patient_name:
            st.error("Please enter both Patient ID and Patient Name.")
        else:
            overall_score, overall_accuracy = calculate_overall(
                balloon_score, path_score, sorting_score,
                balloon_accuracy, path_accuracy, sorting_accuracy,
            )

            # Work out the next session number for this patient.
            existing = data[data["patient_id"] == patient_id]
            next_session_number = int(existing["session_number"].max() + 1) if not existing.empty else 1

            new_row = {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "session_number": next_session_number,
                "session_date": str(session_date),
                "balloon_score": balloon_score,
                "balloon_accuracy": balloon_accuracy,
                "balloon_attempts": 0,
                "balloon_successful_targets": 0,
                "balloon_time_sec": 0,
                "path_score": path_score,
                "path_accuracy": path_accuracy,
                "path_checkpoints_completed": 0,
                "path_time_sec": 0,
                "path_distance_moved": 0,
                "sorting_score": sorting_score,
                "sorting_accuracy": sorting_accuracy,
                "sorting_correct": 0,
                "sorting_wrong": 0,
                "sorting_targets_completed": 0,
                "sorting_time_sec": 0,
                "sorting_distance_moved": 0,
                "is_sample_data": "NO",
            }
            save_new_session(new_row)

            st.success(
                f"Session {next_session_number} saved for {patient_name} "
                f"(Overall Score: {overall_score}, Overall Accuracy: {overall_accuracy}%)."
            )
            st.info("Go to 'Dashboard' or 'Session History' to see the updated data.")