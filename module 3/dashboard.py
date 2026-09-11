"""
MOVENTRA - Module 3: Progress Dashboard
-----------------------------------------
Streamlit dashboard for viewing stroke rehabilitation
progress across:
1. Balloon Reach
2. Follow the Path
3. Object Sorting

IMPORTANT:
This dashboard is a software-based progress monitoring tool only.
It does NOT provide medical diagnosis or clinical decisions.

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

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(THIS_FOLDER, "patient_progress.csv")

CSV_COLUMNS = [
    "patient_id",
    "patient_name",
    "session_number",
    "session_date",

    "balloon_score",
    "balloon_accuracy",
    "balloon_attempts",
    "balloon_successful_targets",
    "balloon_time_sec",

    "path_score",
    "path_accuracy",
    "path_checkpoints_completed",
    "path_time_sec",
    "path_distance_moved",

    "sorting_score",
    "sorting_accuracy",
    "sorting_correct",
    "sorting_wrong",
    "sorting_targets_completed",
    "sorting_time_sec",
    "sorting_distance_moved",

    "is_sample_data",
]


st.set_page_config(
    page_title="MOVENTRA - Progress Dashboard",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------------------------
# 2. DATA HANDLING
# ---------------------------------------------------------------------------

def ensure_csv_exists():
    """Create CSV with correct headers if it does not exist."""
    if not os.path.exists(CSV_PATH):
        empty_df = pd.DataFrame(columns=CSV_COLUMNS)
        empty_df.to_csv(CSV_PATH, index=False)


def load_data():
    """Load patient session data safely."""

    ensure_csv_exists()

    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    # Make sure all expected columns exist
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    if df.empty:
        return df

    # Text columns
    text_columns = [
        "patient_id",
        "patient_name",
        "session_date",
        "is_sample_data",
    ]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)

    # Numeric columns
    numeric_columns = [
        "session_number",

        "balloon_score",
        "balloon_accuracy",
        "balloon_attempts",
        "balloon_successful_targets",
        "balloon_time_sec",

        "path_score",
        "path_accuracy",
        "path_checkpoints_completed",
        "path_time_sec",
        "path_distance_moved",

        "sorting_score",
        "sorting_accuracy",
        "sorting_correct",
        "sorting_wrong",
        "sorting_targets_completed",
        "sorting_time_sec",
        "sorting_distance_moved",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def save_new_session(row_dict):
    """Append a new session to the CSV."""
    df = load_data()

    new_row = pd.DataFrame([row_dict])

    # Make sure new row has all expected columns
    for col in CSV_COLUMNS:
        if col not in new_row.columns:
            new_row[col] = 0

    new_row = new_row[CSV_COLUMNS]

    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(CSV_PATH, index=False)


def calculate_overall(
    balloon_score,
    path_score,
    sorting_score,
    balloon_acc,
    path_acc,
    sorting_acc,
):
    """Calculate simple average score and accuracy."""

    overall_score = round(
        (
            balloon_score
            + path_score
            + sorting_score
        ) / 3,
        1,
    )

    overall_accuracy = round(
        (
            balloon_acc
            + path_acc
            + sorting_acc
        ) / 3,
        1,
    )

    return overall_score, overall_accuracy


# ---------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------

st.sidebar.title("MOVENTRA")
st.sidebar.caption("Module 3 - Progress Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Session History",
        "Add Session",
    ],
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "This dashboard shows software-based progress tracking only. "
    "It is not a medical diagnostic tool."
)


# ---------------------------------------------------------------------------
# 4. LOAD DATA
# ---------------------------------------------------------------------------

data = load_data()


# ---------------------------------------------------------------------------
# 5. HEADER
# ---------------------------------------------------------------------------

st.title("MOVENTRA")
st.subheader("Stroke Rehabilitation Progress Dashboard")


if data.empty:

    st.warning(
        "No session data found yet. "
        "Go to 'Add Session' to add the first session, "
        "or check that patient_progress.csv has data."
    )

    st.stop()


if (
    data["is_sample_data"]
    .astype(str)
    .str.upper()
    .eq("YES")
    .any()
):

    st.info(
        "Some records shown below are DEMONSTRATION SAMPLE DATA for Review 2."
    )


# ---------------------------------------------------------------------------
# 6. PATIENT SELECTION
# ---------------------------------------------------------------------------

def patient_selector(df):
    """Display patient dropdown and return selected patient rows."""

    patient_options = (
        df[["patient_id", "patient_name"]]
        .drop_duplicates()
        .sort_values("patient_id")
    )

    labels = [
        f"{row.patient_id} - {row.patient_name}"
        for row in patient_options.itertuples()
    ]

    choice = st.selectbox(
        "Select Patient",
        labels,
    )

    selected_id = choice.split(" - ")[0]

    patient_rows = (
        df[df["patient_id"] == selected_id]
        .sort_values("session_number")
    )

    return patient_rows, selected_id


# ---------------------------------------------------------------------------
# 7. DASHBOARD PAGE
# ---------------------------------------------------------------------------

if page == "Dashboard":

    patient_rows, patient_id = patient_selector(data)

    latest = patient_rows.iloc[-1]

    # -----------------------------------------------------------------------
    # Patient Information
    # -----------------------------------------------------------------------

    st.markdown("### Patient Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Patient ID",
        latest["patient_id"],
    )

    col2.metric(
        "Patient Name",
        latest["patient_name"],
    )

    col3.metric(
        "Last Session Date",
        str(latest["session_date"]),
    )


    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    overall_score, overall_accuracy = calculate_overall(
        latest["balloon_score"],
        latest["path_score"],
        latest["sorting_score"],

        latest["balloon_accuracy"],
        latest["path_accuracy"],
        latest["sorting_accuracy"],
    )

    total_time_sec = (
        latest["balloon_time_sec"]
        + latest["path_time_sec"]
        + latest["sorting_time_sec"]
    )

    st.markdown("### Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Overall Score",
        f"{overall_score}",
    )

    c2.metric(
        "Overall Accuracy",
        f"{overall_accuracy}%",
    )

    c3.metric(
        "Total Sessions",
        int(patient_rows["session_number"].max()),
    )

    c4.metric(
        "Last Session Exercise Time",
        f"{int(total_time_sec)} sec",
    )


    # -----------------------------------------------------------------------
    # GAME PERFORMANCE
    # -----------------------------------------------------------------------

    st.markdown("### Game Performance (Latest Session)")

    g1, g2, g3 = st.columns(3)


    # -----------------------------------------------------------------------
    # BALLOON REACH
    # -----------------------------------------------------------------------

    with g1:

        st.markdown("#### 🎈 Balloon Reach")

        balloon_table = pd.DataFrame({
            "Metric": [
                "Score",
                "Accuracy",
                "Attempts",
                "Successful Targets",
                "Time (sec)",
            ],

            # IMPORTANT:
            # Every Value is converted to STRING.
            # This prevents Streamlit/PyArrow datatype errors.
            "Value": [
                str(latest["balloon_score"]),
                f"{float(latest['balloon_accuracy']):.1f}%",
                str(int(latest["balloon_attempts"])),
                str(int(latest["balloon_successful_targets"])),
                str(int(latest["balloon_time_sec"])),
            ],
        })

        st.table(
            balloon_table.set_index("Metric")
        )


    # -----------------------------------------------------------------------
    # FOLLOW THE PATH
    # -----------------------------------------------------------------------

    with g2:

        st.markdown("#### 🛤️ Follow the Path")

        path_table = pd.DataFrame({
            "Metric": [
                "Score",
                "Path Accuracy",
                "Checkpoints Completed",
                "Time (sec)",
                "Distance Moved",
            ],

            # Every Value is STRING
            "Value": [
                str(latest["path_score"]),
                f"{float(latest['path_accuracy']):.1f}%",
                str(int(latest["path_checkpoints_completed"])),
                str(int(latest["path_time_sec"])),
                str(int(latest["path_distance_moved"])),
            ],
        })

        st.table(
            path_table.set_index("Metric")
        )


    # -----------------------------------------------------------------------
    # OBJECT SORTING
    # -----------------------------------------------------------------------

    with g3:

        st.markdown("#### 🧩 Object Sorting")

        sorting_table = pd.DataFrame({
            "Metric": [
                "Score",
                "Accuracy",
                "Correct",
                "Wrong",
                "Targets Completed",
                "Time (sec)",
                "Distance Moved",
            ],

            # Every Value is STRING
            "Value": [
                str(latest["sorting_score"]),
                f"{float(latest['sorting_accuracy']):.1f}%",
                str(int(latest["sorting_correct"])),
                str(int(latest["sorting_wrong"])),
                str(int(latest["sorting_targets_completed"])),
                str(int(latest["sorting_time_sec"])),
                str(int(latest["sorting_distance_moved"])),
            ],
        })

        st.table(
            sorting_table.set_index("Metric")
        )


    # -----------------------------------------------------------------------
    # PROGRESS GRAPHS
    # -----------------------------------------------------------------------

    st.markdown("### Progress Across Sessions")


    # -----------------------------------------------------------------------
    # SCORE GRAPH
    # -----------------------------------------------------------------------

    fig1, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(
        patient_rows["session_number"],
        patient_rows["balloon_score"],
        marker="o",
        label="Balloon Reach",
    )

    ax1.plot(
        patient_rows["session_number"],
        patient_rows["path_score"],
        marker="o",
        label="Follow the Path",
    )

    ax1.plot(
        patient_rows["session_number"],
        patient_rows["sorting_score"],
        marker="o",
        label="Object Sorting",
    )

    ax1.set_xlabel("Session Number")
    ax1.set_ylabel("Score")
    ax1.set_title("Score Progress by Game")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    st.pyplot(fig1)


    # -----------------------------------------------------------------------
    # ACCURACY GRAPH
    # -----------------------------------------------------------------------

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(
        patient_rows["session_number"],
        patient_rows["balloon_accuracy"],
        marker="o",
        label="Balloon Reach",
    )

    ax2.plot(
        patient_rows["session_number"],
        patient_rows["path_accuracy"],
        marker="o",
        label="Follow the Path",
    )

    ax2.plot(
        patient_rows["session_number"],
        patient_rows["sorting_accuracy"],
        marker="o",
        label="Object Sorting",
    )

    ax2.set_xlabel("Session Number")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Progress by Game")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)


# ---------------------------------------------------------------------------
# 8. SESSION HISTORY
# ---------------------------------------------------------------------------

elif page == "Session History":

    patient_rows, patient_id = patient_selector(data)

    st.markdown("### Session History")

    history_rows = []

    for _, r in patient_rows.iterrows():

        overall_score, overall_accuracy = calculate_overall(
            r["balloon_score"],
            r["path_score"],
            r["sorting_score"],

            r["balloon_accuracy"],
            r["path_accuracy"],
            r["sorting_accuracy"],
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

    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# 9. ADD SESSION
# ---------------------------------------------------------------------------

elif page == "Add Session":

    st.markdown("### Add New Session")

    st.caption(
        "Enter the results for one rehabilitation session. "
        "This will be saved to patient_progress.csv."
    )


    with st.form(
        "add_session_form",
        clear_on_submit=True,
    ):

        col1, col2 = st.columns(2)


        # -------------------------------------------------------------------
        # Patient Details
        # -------------------------------------------------------------------

        with col1:

            patient_id = st.text_input(
                "Patient ID",
                placeholder="e.g. P003",
            )

            patient_name = st.text_input(
                "Patient Name",
                placeholder="e.g. Suresh Menon",
            )


        with col2:

            session_date = st.date_input(
                "Date"
            )


        # -------------------------------------------------------------------
        # Balloon Reach
        # -------------------------------------------------------------------

        st.markdown("#### 🎈 Balloon Reach")

        b1, b2 = st.columns(2)

        balloon_score = b1.number_input(
            "Balloon Reach Score",
            min_value=0.0,
            step=1.0,
        )

        balloon_accuracy = b2.number_input(
            "Balloon Reach Accuracy (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
        )


        # -------------------------------------------------------------------
        # Follow the Path
        # -------------------------------------------------------------------

        st.markdown("#### 🛤️ Follow the Path")

        p1, p2 = st.columns(2)

        path_score = p1.number_input(
            "Follow the Path Score",
            min_value=0.0,
            step=1.0,
        )

        path_accuracy = p2.number_input(
            "Follow the Path Accuracy (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
        )


        # -------------------------------------------------------------------
        # Object Sorting
        # -------------------------------------------------------------------

        st.markdown("#### 🧩 Object Sorting")

        s1, s2 = st.columns(2)

        sorting_score = s1.number_input(
            "Object Sorting Score",
            min_value=0.0,
            step=1.0,
        )

        sorting_accuracy = s2.number_input(
            "Object Sorting Accuracy (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
        )


        submitted = st.form_submit_button(
            "Save Session"
        )


    # -----------------------------------------------------------------------
    # SAVE NEW SESSION
    # -----------------------------------------------------------------------

    if submitted:

        if not patient_id or not patient_name:

            st.error(
                "Please enter both Patient ID and Patient Name."
            )

        else:

            overall_score, overall_accuracy = calculate_overall(
                balloon_score,
                path_score,
                sorting_score,

                balloon_accuracy,
                path_accuracy,
                sorting_accuracy,
            )


            # Find existing sessions for this patient
            existing = data[
                data["patient_id"].astype(str).str.strip().str.lower()
                == patient_id.strip().lower()
            ]


            if not existing.empty:

                next_session_number = int(
                    existing["session_number"].max()
                ) + 1

            else:

                next_session_number = 1


            new_row = {

                "patient_id": patient_id.strip(),

                "patient_name": patient_name.strip(),

                "session_number": next_session_number,

                "session_date": str(session_date),


                # Balloon Reach
                "balloon_score": balloon_score,
                "balloon_accuracy": balloon_accuracy,
                "balloon_attempts": 0,
                "balloon_successful_targets": 0,
                "balloon_time_sec": 0,


                # Follow the Path
                "path_score": path_score,
                "path_accuracy": path_accuracy,
                "path_checkpoints_completed": 0,
                "path_time_sec": 0,
                "path_distance_moved": 0,


                # Object Sorting
                "sorting_score": sorting_score,
                "sorting_accuracy": sorting_accuracy,
                "sorting_correct": 0,
                "sorting_wrong": 0,
                "sorting_targets_completed": 0,
                "sorting_time_sec": 0,
                "sorting_distance_moved": 0,


                # Sample flag
                "is_sample_data": "NO",
            }


            save_new_session(new_row)


            st.success(
                f"Session {next_session_number} saved for "
                f"{patient_name} "
                f"(Overall Score: {overall_score}, "
                f"Overall Accuracy: {overall_accuracy}%)."
            )

            st.info(
                "Go to 'Dashboard' or 'Session History' "
                "to see the updated data."
            )