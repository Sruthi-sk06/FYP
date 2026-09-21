import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(THIS_FOLDER, "patient_progress.csv")


# ============================================================
# CSV COLUMNS
# ============================================================

CSV_COLUMNS = [
    "patient_id",
    "patient_name",
    "age",
    "training_arm",
    "session_number",
    "session_date",

    # Balloon Reach
    "balloon_score",
    "balloon_accuracy",
    "balloon_attempts",
    "balloon_successful_targets",
    "balloon_time_sec",

    # Follow the Path
    "path_score",
    "path_accuracy",
    "path_checkpoints_completed",
    "path_time_sec",
    "path_distance_moved",

    # Object Sorting
    "sorting_score",
    "sorting_accuracy",
    "sorting_correct",
    "sorting_wrong",
    "sorting_targets_completed",
    "sorting_time_sec",
    "sorting_distance_moved",

    "is_sample_data",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MOVENTRA - Rehabilitation Dashboard",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CSV FUNCTIONS
# ============================================================

def ensure_csv_exists():
    """
    Create the CSV file with the required columns if it
    does not already exist.
    """

    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=CSV_COLUMNS)
        df.to_csv(CSV_PATH, index=False)


def load_data():
    """
    Load patient progress data safely from CSV.
    Missing columns are automatically added.
    """

    ensure_csv_exists()

    try:
        df = pd.read_csv(CSV_PATH)

    except Exception:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    # Add any missing columns
    for column in CSV_COLUMNS:
        if column not in df.columns:

            if column in [
                "patient_id",
                "patient_name",
                "training_arm",
                "session_date",
                "is_sample_data",
            ]:
                df[column] = ""

            else:
                df[column] = 0

    # Keep required column order
    df = df[CSV_COLUMNS]

    # --------------------------------------------------------
    # Text columns
    # --------------------------------------------------------

    text_columns = [
        "patient_id",
        "patient_name",
        "training_arm",
        "session_date",
        "is_sample_data",
    ]

    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "age",
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

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    return df


def save_data(df):
    """
    Save the complete dataframe back to CSV.
    """

    df.to_csv(
        CSV_PATH,
        index=False
    )


# ============================================================
# CALCULATE OVERALL PERFORMANCE
# ============================================================

def calculate_overall(row):
    """
    Calculate overall score and accuracy using the three games.
    """

    scores = [
        float(row.get("balloon_score", 0)),
        float(row.get("path_score", 0)),
        float(row.get("sorting_score", 0)),
    ]

    accuracies = [
        float(row.get("balloon_accuracy", 0)),
        float(row.get("path_accuracy", 0)),
        float(row.get("sorting_accuracy", 0)),
    ]

    overall_score = sum(scores) / len(scores)
    overall_accuracy = sum(accuracies) / len(accuracies)

    return overall_score, overall_accuracy


# ============================================================
# SAVE NEW SESSION
# ============================================================

def save_new_session(
    df,
    patient_id,
    patient_name,
    age,
    training_arm,
    session_date,
    balloon_score,
    balloon_accuracy,
    path_score,
    path_accuracy,
    sorting_score,
    sorting_accuracy,
):
    """
    Add a new patient session to the CSV.
    """

    patient_rows = df[
        df["patient_id"].astype(str).str.strip()
        == str(patient_id).strip()
    ]

    if len(patient_rows) == 0:
        next_session_number = 1

    else:
        next_session_number = int(
            patient_rows["session_number"].max()
        ) + 1

    new_row = {
        "patient_id": str(patient_id).strip(),
        "patient_name": str(patient_name).strip(),
        "age": age,
        "training_arm": training_arm,
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

        "is_sample_data": "NO",
    }

    new_df = pd.DataFrame([new_row])

    df = pd.concat(
        [df, new_df],
        ignore_index=True
    )

    save_data(df)

    return df, next_session_number


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧠 MOVENTRA")

st.sidebar.markdown(
    "### Rehabilitation Monitoring System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Session History",
        "Add Session",
    ]
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">MOVENTRA Rehabilitation Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Camera-based upper-limb rehabilitation monitoring and progress tracking'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    if df.empty:

        st.warning(
            "No patient session data is available yet."
        )

        st.info(
            "Add a session from the 'Add Session' section."
        )

    else:

        # ----------------------------------------------------
        # PATIENT SELECTION
        # ----------------------------------------------------

        patient_ids = sorted(
            df["patient_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_patient = st.selectbox(
            "Select Patient",
            patient_ids
        )

        patient_df = df[
            df["patient_id"].astype(str)
            == str(selected_patient)
        ].copy()

        patient_df = patient_df.sort_values(
            "session_number"
        )

        latest = patient_df.iloc[-1]

        # ----------------------------------------------------
        # PATIENT INFORMATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Patient Information</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Patient ID",
                latest["patient_id"]
            )

        with col2:
            st.metric(
                "Name",
                latest["patient_name"]
            )

        with col3:
            age_value = int(latest["age"])

            st.metric(
                "Age",
                age_value
            )

        with col4:
            st.metric(
                "Training Arm",
                latest["training_arm"]
            )

        with col5:
            st.metric(
                "Sessions",
                len(patient_df)
            )

        # ----------------------------------------------------
        # LATEST SESSION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Latest Session</div>',
            unsafe_allow_html=True
        )

        overall_score, overall_accuracy = calculate_overall(
            latest
        )

        total_exercise_time = (
            float(latest["balloon_time_sec"])
            + float(latest["path_time_sec"])
            + float(latest["sorting_time_sec"])
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Overall Score",
                f"{overall_score:.1f}"
            )

        with col2:
            st.metric(
                "Overall Accuracy",
                f"{overall_accuracy:.1f}%"
            )

        with col3:
            st.metric(
                "Session",
                int(latest["session_number"])
            )

        with col4:
            st.metric(
                "Exercise Time",
                f"{total_exercise_time:.1f} sec"
            )

        # ----------------------------------------------------
        # LATEST GAME PERFORMANCE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Latest Game Performance</div>',
            unsafe_allow_html=True
        )

        game_col1, game_col2, game_col3 = st.columns(3)

        # ----------------------------------------------------
        # BALLOON REACH
        # ----------------------------------------------------

        with game_col1:

            st.subheader("🎈 Balloon Reach")

            st.metric(
                "Score",
                f"{float(latest['balloon_score']):.1f}"
            )

            st.metric(
                "Accuracy",
                f"{float(latest['balloon_accuracy']):.1f}%"
            )

            st.metric(
                "Attempts",
                int(latest["balloon_attempts"])
            )

            st.metric(
                "Successful Targets",
                int(latest["balloon_successful_targets"])
            )

            st.metric(
                "Time",
                f"{float(latest['balloon_time_sec']):.1f} sec"
            )

        # ----------------------------------------------------
        # FOLLOW THE PATH
        # ----------------------------------------------------

        with game_col2:

            st.subheader("🛤️ Follow the Path")

            st.metric(
                "Score",
                f"{float(latest['path_score']):.1f}"
            )

            st.metric(
                "Accuracy",
                f"{float(latest['path_accuracy']):.1f}%"
            )

            st.metric(
                "Checkpoints",
                int(latest["path_checkpoints_completed"])
            )

            st.metric(
                "Time",
                f"{float(latest['path_time_sec']):.1f} sec"
            )

            st.metric(
                "Distance",
                f"{float(latest['path_distance_moved']):.1f}"
            )

        # ----------------------------------------------------
        # OBJECT SORTING
        # ----------------------------------------------------

        with game_col3:

            st.subheader("🔵 Object Sorting")

            st.metric(
                "Score",
                f"{float(latest['sorting_score']):.1f}"
            )

            st.metric(
                "Accuracy",
                f"{float(latest['sorting_accuracy']):.1f}%"
            )

            st.metric(
                "Correct",
                int(latest["sorting_correct"])
            )

            st.metric(
                "Wrong",
                int(latest["sorting_wrong"])
            )

            st.metric(
                "Targets Completed",
                int(latest["sorting_targets_completed"])
            )

            st.metric(
                "Time",
                f"{float(latest['sorting_time_sec']):.1f} sec"
            )

        # ----------------------------------------------------
        # PERFORMANCE TREND
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Performance Progress</div>',
            unsafe_allow_html=True
        )

        graph_df = patient_df.copy()

        graph_df["overall_score"] = (
            graph_df[
                [
                    "balloon_score",
                    "path_score",
                    "sorting_score",
                ]
            ]
            .astype(float)
            .mean(axis=1)
        )

        graph_df["overall_accuracy"] = (
            graph_df[
                [
                    "balloon_accuracy",
                    "path_accuracy",
                    "sorting_accuracy",
                ]
            ]
            .astype(float)
            .mean(axis=1)
        )

        # ----------------------------------------------------
        # SCORE GRAPH
        # ----------------------------------------------------

        st.subheader("Score Across Sessions")

        fig1, ax1 = plt.subplots()

        ax1.plot(
            graph_df["session_number"],
            graph_df["balloon_score"],
            marker="o",
            label="Balloon Reach"
        )

        ax1.plot(
            graph_df["session_number"],
            graph_df["path_score"],
            marker="o",
            label="Follow the Path"
        )

        ax1.plot(
            graph_df["session_number"],
            graph_df["sorting_score"],
            marker="o",
            label="Object Sorting"
        )

        ax1.plot(
            graph_df["session_number"],
            graph_df["overall_score"],
            marker="o",
            linestyle="--",
            label="Overall Score"
        )

        ax1.set_xlabel("Session Number")
        ax1.set_ylabel("Score")
        ax1.set_title("Game Score Progress")

        ax1.legend()
        ax1.grid(True, alpha=0.3)

        st.pyplot(fig1)

        plt.close(fig1)

        # ----------------------------------------------------
        # ACCURACY GRAPH
        # ----------------------------------------------------

        st.subheader("Accuracy Across Sessions")

        fig2, ax2 = plt.subplots()

        ax2.plot(
            graph_df["session_number"],
            graph_df["balloon_accuracy"],
            marker="o",
            label="Balloon Reach"
        )

        ax2.plot(
            graph_df["session_number"],
            graph_df["path_accuracy"],
            marker="o",
            label="Follow the Path"
        )

        ax2.plot(
            graph_df["session_number"],
            graph_df["sorting_accuracy"],
            marker="o",
            label="Object Sorting"
        )

        ax2.plot(
            graph_df["session_number"],
            graph_df["overall_accuracy"],
            marker="o",
            linestyle="--",
            label="Overall Accuracy"
        )

        ax2.set_xlabel("Session Number")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Game Accuracy Progress")

        ax2.legend()
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig2)

        plt.close(fig2)


# ============================================================
# SESSION HISTORY
# ============================================================

elif page == "Session History":

    st.subheader("📋 Session History")

    if df.empty:

        st.info(
            "No session records are available."
        )

    else:

        patient_ids = sorted(
            df["patient_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_patient = st.selectbox(
            "Select Patient",
            patient_ids,
            key="history_patient"
        )

        history_df = df[
            df["patient_id"].astype(str)
            == str(selected_patient)
        ].copy()

        history_df = history_df.sort_values(
            "session_number"
        )

        # Calculate overall values
        overall_scores = []
        overall_accuracies = []

        for _, row in history_df.iterrows():

            score, accuracy = calculate_overall(row)

            overall_scores.append(score)
            overall_accuracies.append(accuracy)

        history_df["overall_score"] = overall_scores
        history_df["overall_accuracy"] = overall_accuracies

        display_df = history_df[
            [
                "session_number",
                "session_date",
                "training_arm",

                "balloon_score",
                "path_score",
                "sorting_score",

                "overall_score",
                "overall_accuracy",
            ]
        ].copy()

        display_df.columns = [
            "Session",
            "Date",
            "Training Arm",
            "Balloon Score",
            "Path Score",
            "Sorting Score",
            "Overall Score",
            "Overall Accuracy",
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ADD SESSION
# ============================================================

elif page == "Add Session":

    st.subheader("➕ Add Patient Session")

    st.write(
        "Enter the patient's details and session performance."
    )

    # --------------------------------------------------------
    # PATIENT DETAILS
    # --------------------------------------------------------

    st.markdown("### Patient Details")

    col1, col2 = st.columns(2)

    with col1:

        patient_id = st.text_input(
            "Patient ID",
            placeholder="e.g. P001"
        )

        patient_name = st.text_input(
            "Patient Name",
            placeholder="e.g. Suresh Menon"
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1
        )

    with col2:

        training_arm = st.selectbox(
            "Training Arm",
            [
                "LEFT",
                "RIGHT"
            ]
        )

        session_date = st.date_input(
            "Session Date"
        )

    # --------------------------------------------------------
    # BALLOON REACH
    # --------------------------------------------------------

    st.markdown("### 🎈 Balloon Reach")

    col1, col2 = st.columns(2)

    with col1:

        balloon_score = st.number_input(
            "Balloon Reach Score",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    with col2:

        balloon_accuracy = st.number_input(
            "Balloon Reach Accuracy (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

    # --------------------------------------------------------
    # FOLLOW THE PATH
    # --------------------------------------------------------

    st.markdown("### 🛤️ Follow the Path")

    col1, col2 = st.columns(2)

    with col1:

        path_score = st.number_input(
            "Follow the Path Score",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    with col2:

        path_accuracy = st.number_input(
            "Follow the Path Accuracy (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

    # --------------------------------------------------------
    # OBJECT SORTING
    # --------------------------------------------------------

    st.markdown("### 🔵 Object Sorting")

    col1, col2 = st.columns(2)

    with col1:

        sorting_score = st.number_input(
            "Object Sorting Score",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    with col2:

        sorting_accuracy = st.number_input(
            "Object Sorting Accuracy (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    st.markdown("###")

    if st.button(
        "💾 Save Session",
        type="primary",
        use_container_width=True
    ):

        if not patient_id.strip():

            st.error(
                "Please enter a Patient ID."
            )

        elif not patient_name.strip():

            st.error(
                "Please enter the Patient Name."
            )

        else:

            df, session_number = save_new_session(
                df=df,
                patient_id=patient_id,
                patient_name=patient_name,
                age=age,
                training_arm=training_arm,
                session_date=session_date,
                balloon_score=balloon_score,
                balloon_accuracy=balloon_accuracy,
                path_score=path_score,
                path_accuracy=path_accuracy,
                sorting_score=sorting_score,
                sorting_accuracy=sorting_accuracy,
            )

            st.success(
                f"Session {session_number} saved successfully "
                f"for {patient_name.strip()}."
            )

            st.info(
                f"Training arm: {training_arm}"
            )

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "MOVENTRA | Camera-Based Upper-Limb Rehabilitation"
)

st.sidebar.caption(
    "For rehabilitation monitoring and research purposes."
)