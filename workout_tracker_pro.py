# =========================================================
# WORKOUT TRACKER PRO — CLOUD EDITION (2025)
# → Fully converted to Supabase (PostgreSQL)
# → Multi-user, Admin Panel, EMG Load, Week-vs-Week, Backup
# → Deploy instantly: https://share.streamlit.io
# =========================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from passlib.hash import pbkdf2_sha256
from streamlit_option_menu import option_menu
from io import BytesIO
from supabase import create_client, Client

# ====================== PAGE CONFIG & SUPABASE ======================
st.set_page_config(page_title="Workout Tracker Pro", layout="wide")

@st.cache_resource
def get_supabase() -> Client:
    # create_client takes positional args (url, key), not keyword args
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_ANON_KEY"]
    )

@st.cache_resource
def get_supabase_admin() -> Client:
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
    )

supabase = get_supabase()
supabase_admin = get_supabase_admin()

# ====================== AUTH & SUPER ADMIN ======================
def hash_password(p): return pbkdf2_sha256.hash(p)
def verify_password(p, h): return pbkdf2_sha256.verify(p, h)

def login(username: str, password: str):
    res = supabase.table("users").select("*").eq("username", username).execute()
    if res.data and verify_password(password, res.data[0]["password"]):
        u = res.data[0]
        return {"id": str(u["id"]), "username": u["username"], "is_admin": u.get("is_admin", False)}
    return None

def signup(username: str, password: str):
    try:
        supabase.table("users").insert({"username": username, "password": hash_password(password)}).execute()
        return True
    except:
        return False

def create_super_admin():
    res = supabase.table("users").select("id").eq("username", "admin").execute()
    if not res.data:
        supabase_admin.table("users").insert({
            "username": "admin",
            "password": hash_password("ChangeMe123!Now2025"),
            "is_admin": True
        }).execute()
        st.success("SUPER ADMIN CREATED → CHANGE PASSWORD NOW!")

create_super_admin()

# ====================== CORE DB FUNCTIONS ======================
def add_exercise_entry(uid, date, exercise, sets, reps, weight):
    supabase.table("exercises").insert({
        "user_id": uid,
        "date": date.strftime("%Y-%m-%d"),
        "exercise": exercise,
        "sets": sets,
        "reps": reps,
        "weight": weight
    }).execute()

def get_user_exercises(uid):
    res = supabase.table("exercises").select("*").eq("user_id", uid).order("date", desc=True).execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

def calculate_1rm(w, r):
    return round(w * 36 / (37 - r), 1) if 1 <= r <= 10 else round(w, 1)

def add_emg_load(df):
    if df.empty: return df
    res = supabase.table("emg_reference").select("*").execute()
    emg_df = pd.DataFrame(res.data)
    if emg_df.empty: return df
    df['exercise_lower'] = df['exercise'].str.lower().str.strip()
    emg_df['exercise_lower'] = emg_df['exercise'].str.lower().str.strip()
    merged = df.merge(emg_df, on='exercise_lower', how='left')
    merged['volume'] = merged['sets'] * merged['reps'] * merged['weight']
    volume_safe = merged['volume'].replace(0, 1)
    muscle_cols = [c for c in emg_df.columns if c not in ['exercise', 'exercise_lower']]
    for m in muscle_cols:
        if m in merged.columns:
            merged[f"{m}_eff"] = merged[m].fillna(0) * volume_safe / 10.0
    return merged.drop(columns=['exercise_lower'], errors='ignore')

def download_template(cols, name):
    df = pd.DataFrame(columns=cols)
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    return output.getvalue()

# ====================== ADMIN & IMPORT FUNCTIONS ======================
def get_all_users():
    res = supabase_admin.table("users").select("id, username, is_admin").execute()
    return pd.DataFrame(res.data)

def delete_user(user_id):
    supabase_admin.table("exercises").delete().eq("user_id", user_id).execute()
    supabase_admin.table("users").delete().eq("id", user_id).execute()

def toggle_admin(user_id, current):
    supabase_admin.table("users").update({"is_admin": not current}).eq("id", user_id).execute()

def log_action(admin, action, target="", details=""):
    supabase_admin.table("admin_log").insert({
        "admin": admin, "action": action, "target": target, "details": details
    }).execute()

def import_workout_logs(uid, df):
    if df.empty:
        st.error("File is empty!")
        return

    # 1. Normalize column names once
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.strip()
    col_lower = {col.strip().lower(): col.strip() for col in df.columns}

    # 2. Smart column detection (very forgiving)
    date_col = exercise_col = sets_col = reps_col = weight_col = None

    for lower, orig in col_lower.items():
        if any(x in lower for x in ['date', 'day', 'time']):
            date_col = orig
        elif any(x in lower for x in ['exercise', 'exer', 'movement', 'lift']):
            exercise_col = orig
        elif 'set' in lower:
            sets_col = orig
        elif 'rep' in lower:
            reps_col = orig
        elif any(x in lower for x in ['weight', 'wgt', 'kg', 'load', 'lbs']):
            weight_col = orig

    missing = []
    if not date_col: missing.append("Date")
    if not exercise_col: missing.append("Exercise")
    if not sets_col: missing.append("Sets")
    if not reps_col: missing.append("Reps")
    if not weight_col: missing.append("Weight (kg)")

    if missing:
        st.error(f"Could not find columns: {', '.join(missing)}")
        st.info("Detected columns: " + ", ".join(original_cols))
        return

    # 3. Keep only needed columns and rename
    df = df[[date_col, exercise_col, sets_col, reps_col, weight_col]].copy()
    df.columns = ['date', 'exercise', 'sets', 'reps', 'weight']

    # 4. Clean numeric values (removes "kg", "lbs", "reps", etc.)
    def clean_number(x):
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        s = ''.join(c for c in s if c.isdigit() or c in '.-')
        try:
            return float(s) if s else None
        except:
            return None

    df['sets'] = df['sets'].apply(clean_number)
    df['reps'] = df['reps'].apply(clean_number)
    df['weight'] = df['weight'].apply(clean_number)

    # 5. Parse dates (very forgiving)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    # 6. Drop completely invalid rows
    before = len(df)
    df = df.dropna(subset=['date', 'sets', 'reps', 'weight', 'exercise'])
    df = df[(df['sets'] > 0) & (df['reps'] > 0) & (df['weight'] >= 0)]
    after = len(df)

    skipped = before - after
    imported = 0
    for _, row in df.iterrows():
        try:
            # THIS IS THE ONLY LINE THAT CHANGED — use supabase_admin to bypass RLS
            supabase_admin.table("exercises").insert({
                "user_id": uid,
                "date": row['date'].date().isoformat(),
                "exercise": str(row['exercise']).strip(),
                "sets": int(row['sets']),
                "reps": int(row['reps']),
                "weight": float(row['weight'])
            }).execute()
            imported += 1
        except Exception as e:
            st.warning(f"Row skipped: {row.to_dict()} → {e}")

    st.success(f"Imported {imported} workouts successfully!")
        
def import_exercise_reference(df):
    req = ['Exercise', 'Group', 'Primary', 'Secondary']
    if not all(col in df.columns for col in req):
        st.error(f"Missing columns: {set(req) - set(df.columns)}"); return
    for _, row in df.iterrows():
        e = str(row['Exercise']).strip()
        g = str(row['Group']).strip() if pd.notna(row['Group']) else None
        p = str(row['Primary']).strip() if pd.notna(row['Primary']) else None
        s = str(row['Secondary']).strip() if pd.notna(row['Secondary']) else None
        supabase_admin.table("exercise_reference").upsert({
            "exercise": e, "group_name": g, "primary_muscle": p, "secondary_muscle": s
        }, on_conflict="exercise").execute()
    st.success("Exercise reference updated!")

def import_emg_reference(df):
    if df.empty:
        st.error("File is empty!")
        return

    imported = 0
    for _, row in df.iterrows():
        ex = str(row['Exercise']).strip()
        if not ex or ex == 'nan':
            continue

        data = {"exercise": ex}
        for col in df.columns:
            if col == "Exercise":
                continue
            try:
                data[col] = float(row[col]) if pd.notna(row[col]) else 0.0
            except:
                data[col] = 0.0

        try:
            supabase_admin.table("emg_reference").upsert(data, on_conflict="exercise").execute()
            imported += 1
        except Exception as e:
            st.warning(f"Failed {ex}: {e}")

    st.success(f"EMG DATA FINALLY LOADED — {imported} exercises imported!")
    
# ====================== MAIN UI ======================
st.markdown("<h1 style='text-align: center; color: #1e40af;'>Workout Tracker Pro</h1>", unsafe_allow_html=True)

if "user" not in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                user = login(u, p)
                if user: st.session_state.user = user; st.rerun()
                else: st.error("Wrong credentials")
    with col2:
        with st.form("signup"):
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            if st.form_submit_button("Sign Up"):
                if signup(nu, np):
                    st.success("Created! Logging in...")
                    st.session_state.user = login(nu, np)
                    st.rerun()
                else: st.error("Username taken")
else:
    uid = st.session_state.user["id"]
    uname = st.session_state.user["username"]
    is_admin = st.session_state.user.get("is_admin", False)

    with st.sidebar:
        st.success(f"**{uname}**")
        if is_admin: st.markdown("**Admin**")
        if st.button("Logout"): st.session_state.clear(); st.rerun()
        if st.button("Change Password"): st.session_state.pw_change = True

        if st.session_state.get("pw_change"):
            with st.form("pw_form"):
                old = st.text_input("Old", type="password")
                new1 = st.text_input("New", type="password")
                new2 = st.text_input("Confirm", type="password")
                if st.form_submit_button("Update"):
                    if new1 != new2: st.error("Passwords don't match")
                    elif len(new1) < 8: st.error("Too short")
                    else:
                        user_res = supabase.table("users").select("password").eq("id", uid).execute()
                        if verify_password(old, user_res.data[0]["password"]):
                            supabase.table("users").update({"password": hash_password(new1)}).eq("id", uid).execute()
                            st.success("Password changed!")
                            st.session_state.pw_change = False
                            st.rerun()
                        else:
                            st.error("Wrong old password")

        menu = ["Log Workout", "Dashboard", "PRs", "Data Import"]
        icons = ["plus", "graph-up", "trophy", "upload"]
        if is_admin:
            menu += ["EMG Editor", "Admin Panel"]
            icons += ["activity", "shield-lock"]
        selected = option_menu(None, menu, icons=icons, menu_icon="cast", default_index=0)

    # ====================== LOG WORKOUT ======================
    if selected == "Log Workout":
        st.markdown("### Log Workout")
        res = supabase.table("exercise_reference").select("exercise, group_name").execute()
        ref_df = pd.DataFrame(res.data)
        options = ["Select..."]
        if not ref_df.empty:
            ref_df['group_name'] = ref_df['group_name'].fillna("Other")
            ref_df['sort'] = ref_df['group_name'] + " → " + ref_df['exercise']
            options += ref_df.sort_values('sort')['exercise'].tolist()
        options += ["+ Add New"]

        ex = st.selectbox("Exercise", options)
        if ex == "+ Add New":
            with st.expander("New Exercise", expanded=True):
                with st.form("new_ex"):
                    name = st.text_input("Name")
                    group = st.selectbox("Group", ["Chest","Back","Shoulders","Arms","Legs","Core","Other"])
                    if st.form_submit_button("Create"):
                        supabase_admin.table("exercise_reference").insert({"exercise": name, "group_name": group}).execute()
                        st.success("Created!"); st.rerun()

        if ex and ex not in ["Select...", "+ Add New"]:
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: date = st.date_input("Date", datetime.today())
            with c2: sets = st.number_input("Sets",1,50,3)
            with c3: reps = st.number_input("Reps",1,100,8)
            with c4: weight = st.number_input("Weight (kg)",0.0,1000.0,80.0,0.5)
            with c5:
                if st.button("LOG", type="primary"):
                    add_exercise_entry(uid, date, ex, sets, reps, weight)
                    st.success("Logged!"); st.balloons(); st.rerun()

        st.markdown("### Recent Workouts")
        recent = get_user_exercises(uid).head(50)
        if not recent.empty:
            recent['date'] = pd.to_datetime(recent['date']).dt.strftime("%d %b %Y")
            recent['vol'] = recent['sets'] * recent['reps'] * recent['weight']
            disp = recent[['date','exercise','sets','reps','weight','vol']].copy()
            disp.columns = ['Date','Exercise','Sets','Reps','Weight','Volume']
            sel = st.dataframe(disp, use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun")
            if sel and sel["selection"]["rows"]:
                row_id = recent.iloc[sel["selection"]["rows"][0]]["id"]
                if st.button("Delete Entry", type="secondary"):
                    supabase.table("exercises").delete().eq("id", row_id).execute()
                    st.rerun()

    # ====================== DASHBOARD ======================
    elif selected == "Dashboard":
        st.markdown("### Performance Dashboard")
        df = get_user_exercises(uid)
        if df.empty:
            st.info("No workouts logged yet — start crushing it!")
        else:
            df['date'] = pd.to_datetime(df['date'])
            df['volume'] = df['sets'] * df['reps'] * df['weight']

            ref = supabase.table("exercise_reference").select("exercise, group_name, primary_muscle").execute()
            ref_df = pd.DataFrame(ref.data)
            if not ref_df.empty:
                df = df.merge(ref_df[['exercise', 'group_name', 'primary_muscle']], on='exercise', how='left')
                df['group_name'] = df['group_name'].fillna('Other')
                df['primary_muscle'] = df['primary_muscle'].fillna('')
            else:
                df['group_name'] = 'Other'; df['primary_muscle'] = ''

            df['year'] = df['date'].dt.year
            df['week'] = df['date'].dt.isocalendar().week
            df['year_week'] = df['year'].astype(str) + "-W" + df['week'].astype(str).str.zfill(2)

            with st.expander("Filters", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", df['date'].min().date())
                    end_date = st.date_input("End Date", df['date'].max().date())
                with col2:
                    years = st.multiselect("Years", sorted(df['year'].unique()), default=sorted(df['year'].unique()))
                    weeks = st.multiselect("Weeks", sorted(df['week'].unique()), default=sorted(df['week'].unique()))
                groups = st.multiselect("Muscle Groups", sorted(df['group_name'].unique()), default=sorted(df['group_name'].unique()))

            mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date) & \
                   df['year'].isin(years) & df['week'].isin(weeks) & df['group_name'].isin(groups)
            df_f = df[mask].copy()

            if df_f.empty:
                st.warning("No data in selected period")
            else:
                c1,c2,c3 = st.columns(3)
                c1.metric("Total Volume", f"{df_f['volume'].sum():,.0f} kg")
                c2.metric("Sessions", df_f['date'].dt.date.nunique())
                c3.metric("Avg per Session", f"{df_f['volume'].sum()/df_f['date'].dt.date.nunique():,.0f} kg")

                tab1,tab2,tab3,tab4 = st.tabs(["Groups","Muscle Heatmap","EMG Load","Week vs Week"])

                with tab1:
                    fig = px.area(df_f.groupby(['year_week','group_name'])['volume'].sum().reset_index(),
                                 x='year_week', y='volume', color='group_name', height=700)
                    fig.update_layout(title="Volume by Muscle Group", legend_title="Group")
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    exploded = []
                    for _, r in df_f.iterrows():
                        muscles = [m.strip() for m in str(r['primary_muscle']).split(",") if m.strip()]
                        vol = r['volume'] / len(muscles) if muscles else 0
                        for m in muscles:
                            exploded.append({"muscle": m, "group": r['group_name'] or "Unknown", "volume": vol})
                    heat = pd.DataFrame(exploded)
                    if not heat.empty:
                        pivot = heat.pivot_table(index='muscle', columns='group', values='volume', aggfunc='sum', fill_value=0)
                        fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                                                       colorscale="OrRd", text=pivot.values,
                                                       texttemplate="%{text:,.0f}", textfont={"size":12}))
                        fig.update_layout(height=700, title="Muscle Heatmap (kg)")
                        st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    df_e = add_emg_load(df_f.copy())
                    cols = [c for c in df_e.columns if c.endswith('_eff')]
                    if not cols or df_e[cols].sum().sum() == 0:
                        st.warning("No EMG data available")
                    else:
                        load = df_e[cols].sum().reset_index()
                        load.columns = ['muscle_raw', 'load']
                        load['muscle'] = load['muscle_raw'].str.replace('_eff','',regex=False).str.replace('_',' ',regex=False).str.title()
                        load = load[load['load']>0].sort_values('load', ascending=False)
                        fig = px.bar(load, x='load', y='muscle', orientation='h', color='load',
                                    color_continuous_scale='Viridis', text=load['load'].round(0).astype(int).astype(str)+"k",
                                    height=max(1000, len(load)*35))
                        fig.update_layout(title="EMG Load Distribution", showlegend=False,
                                          xaxis_title="EMG Load (×1000)", margin=dict(l=280))
                        st.plotly_chart(fig, use_container_width=True)

                with tab4:
                    st.markdown("### Week-vs-Week Muscle Volume")
                    week_options = sorted(df_f['year_week'].unique(), reverse=True)
                    if len(week_options) < 2:
                        st.info("Need at least 2 weeks of data")
                    else:
                        colA, colB = st.columns(2)
                        with colA: week_new = st.selectbox("Recent Week", week_options, index=0)
                        with colB: week_old = st.selectbox("Previous Week", [w for w in week_options if w != week_new], index=0)

                        new_data = df_f[df_f['year_week'] == week_new].copy()
                        old_data = df_f[df_f['year_week'] == week_old].copy()

                        def get_muscles(row):
                            m = str(row['primary_muscle'])
                            if pd.isna(m) or m in ['nan', '']: return []
                            return [x.strip() for x in m.split(",") if x.strip()]

                        def build_volume(data):
                            records = []
                            for _, r in data.iterrows():
                                muscles = get_muscles(r)
                                if not muscles: continue
                                vol_per = r['volume'] / len(muscles)
                                for m in muscles:
                                    records.append({"muscle": m, "volume": vol_per})
                            return pd.DataFrame(records).groupby('muscle')['volume'].sum() if records else pd.Series()

                        new_vol = build_volume(new_data)
                        old_vol = build_volume(old_data)

                        if new_vol.empty or old_vol.empty:
                            st.warning("Not enough data")
                        else:
                            combined = pd.concat([new_vol, old_vol], axis=1).fillna(0)
                            combined.columns = ['New', 'Old']
                            combined['Change %'] = ((combined['New'] - combined['Old']) / combined['Old'].replace(0, 1) * 100).round(1)
                            combined = combined.sort_values('New', ascending=False).head(20)

                            fig = go.Figure()
                            fig.add_trace(go.Bar(y=combined.index, x=combined['Old'], name=f"{week_old}", orientation='h',
                                                marker=dict(color='#94a3b8'), text=[f"<b>{int(x):,} kg</b>" for x in combined['Old']],
                                                textposition='inside'))
                            fig.add_trace(go.Bar(y=combined.index, x=combined['New'], name=f"{week_new}", orientation='h',
                                                marker=dict(color='#1e40af', line=dict(color='#1e3a8a', width=2.5)),
                                                text=[f"<b>{int(x):,} kg</b>" for x in combined['New']], textposition='inside'))

                            for i, (muscle, row) in enumerate(combined.iterrows()):
                                change = row['Change %']
                                sign = "+" if change >= 0 else ""
                                color = "#16a34a" if change >= 0 else "#dc2626"
                                fig.add_annotation(x=row['New'], y=muscle, text=f"<b>{sign}{change}%</b>", showarrow=False,
                                    xanchor="left", xshift=15, font=dict(size=12, color=color, family="Arial Black"),
                                    bgcolor="rgba(255,255,255,0.98)", bordercolor=color, borderwidth=2.5, borderpad=6)

                            max_x = combined['New'].max() * 1.58
                            for i in range(len(combined)):
                                if i % 2 == 0:
                                    fig.add_shape(type="rect", x0=0, x1=max_x, y0=i-0.5, y1=i+0.5,
                                        fillcolor="rgba(30,64,175,0.04)", line_width=0, layer="below")

                            fig.update_layout(height=880, title=f"<b>{week_new}</b> vs <b>{week_old}</b>",
                                barmode='overlay', bargap=0.38, xaxis=dict(range=[0, max_x]), margin=dict(l=220, r=200, t=120, b=90))
                            st.plotly_chart(fig, use_container_width=True)

    # ====================== PRs ======================
    elif selected == "PRs":
        st.markdown("### Personal Records")
        df = get_user_exercises(uid).copy()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['1rm'] = df.apply(lambda x: calculate_1rm(x['weight'], x['reps']), axis=1)
            exercises = st.multiselect("Exercises", sorted(df['exercise'].unique()), default=sorted(df['exercise'].unique())[:5])
            if exercises:
                fig = go.Figure()
                for ex in exercises:
                    data = df[df['exercise']==ex].sort_values('date')
                    fig.add_scatter(x=data['date'], y=data['1rm'], mode='lines+markers', name=ex)
                fig.update_layout(height=700, title="1RM Progression")
                st.plotly_chart(fig, use_container_width=True)

            prs = df.loc[df.groupby('exercise')['1rm'].idxmax()].copy()
            prs['date'] = prs['date'].dt.strftime("%d %b %Y")
            prs = prs[['exercise','1rm','weight','reps','sets','date']].round(1)
            prs.columns = ['Exercise','Est 1RM','Weight','Reps','Sets','Date']
            prs = prs.sort_values('Est 1RM', ascending=False)
            st.dataframe(prs, use_container_width=True, hide_index=True)

    # ====================== DATA IMPORT ======================
    elif selected == "Data Import":
        tabs = st.tabs(["Workouts"] + (["Exercise Reference"] if is_admin else []))
        with tabs[0]:
            st.download_button("Download Template", download_template(['Date','Exercise','Sets','Reps','Weight (kg)'],"workouts.xlsx"), "workouts.xlsx")
            file = st.file_uploader("Upload workouts", type="xlsx")
            if file and st.button("Import Workouts", type="primary"):
                df = pd.read_excel(file)
                import_workout_logs(uid, df)
        if is_admin and len(tabs) > 1:
            with tabs[1]:
                st.download_button("Download Reference Template", download_template(['Exercise','Group','Primary','Secondary'],"reference.xlsx"), "reference.xlsx")
                file = st.file_uploader("Upload Exercise Reference", type="xlsx", key="ref")
                if file and st.button("IMPORT REFERENCE", type="primary"):
                    df = pd.read_excel(file)
                    import_exercise_reference(df)

    # ====================== EMG EDITOR ======================
    elif selected == "EMG Editor" and is_admin:
        st.download_button("Download EMG Template", download_template(['exercise','chest','back','quads','hamstrings','glutes','delts_front','delts_side','triceps','biceps','calves'], "EMG_DATA.xlsx"), "EMG_DATA.xlsx")
        file = st.file_uploader("Upload EMG Data", type="xlsx")
        if file:
            df = pd.read_excel(file)
            import_emg_reference(df)

    # ====================== ADMIN PANEL ======================
    elif selected == "Admin Panel" and is_admin:
        total_users = len(get_all_users())
        total_workouts = supabase_admin.table("exercises").select("id", count="exact").execute().count
        total_volume = supabase_admin.table("exercises").select("sets,reps,weight").execute()
        vol_sum = sum(r["sets"]*r["reps"]*r["weight"] for r in total_volume.data) if total_volume.data else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Users", total_users)
        c2.metric("Total Workouts", f"{total_workouts:,}")
        c3.metric("Global Volume", f"{vol_sum/1_000_000:.2f}M kg")
        c4.metric("Active (30d)", "Soon")

        st.markdown("---")
        users = get_all_users()
        search = st.text_input("Search users")
        if search:
            users = users[users['username'].str.contains(search, case=False, na=False)]

        for _, row in users.iterrows():
            with st.expander(f"**{row['username']}** • ID: {row['id']} • {'Admin' if row['is_admin'] else 'User'}", expanded=False):
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 3])
                with col1:
                    if st.button("Toggle Admin", key=f"tog_{row['id']}"):
                        toggle_admin(row['id'], row['is_admin'])
                        log_action(uname, "Toggle admin", row['username'])
                        st.rerun()
                with col2:
                    if st.button("Login As", key=f"imp_{row['id']}", type="primary"):
                        st.session_state.user = {"id": str(row['id']), "username": row['username'], "is_admin": True}
                        log_action(uname, "Impersonate", row['username'])
                        st.rerun()
                with col3:
                    if st.button("View Stats", key=f"stats_{row['id']}"):
                        st.session_state.viewing_stats = row['id']
                with col4:
                    if st.button("DELETE USER", key=f"del_{row['id']}", type="secondary"):
                        if row['id'] != uid and st.checkbox("Confirm delete", key=f"confirm_{row['id']}"):
                            delete_user(row['id'])
                            log_action(uname, "Delete user", row['username'])
                            st.rerun()
                with col5:
                    user_data = get_user_exercises(row['id'])
                    if not user_data.empty:
                        vol = (user_data['sets']*user_data['reps']*user_data['weight']).sum()
                        st.caption(f"{len(user_data):,} workouts • {vol/1000:,.0f}k kg")

                if st.session_state.get("viewing_stats") == row['id']:
                    dfu = get_user_exercises(row['id'])
                    if not dfu.empty:
                        dfu['date'] = pd.to_datetime(dfu['date'])
                        dfu['volume'] = dfu['sets'] * dfu['reps'] * dfu['weight']
                        st.subheader(f"Stats for {row['username']}")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Workouts", len(dfu))
                        mc2.metric("Total Volume", f"{dfu['volume'].sum():,.0f} kg")
                        mc3.metric("First", dfu['date'].min().strftime("%d %b %Y"))
                        fig = px.line(dfu.groupby(dfu['date'].dt.date)['volume'].sum().reset_index(), x='date', y='volume')
                        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Admin Actions Log", expanded=False):
            log_res = supabase_admin.table("admin_log").select("*").order("id", desc=True).limit(50).execute()
            log_df = pd.DataFrame(log_res.data)
            if not log_df.empty:
                st.dataframe(log_df[['timestamp','admin','action','target','details']], use_container_width=True, hide_index=True)

st.caption("© 2025 Workout Tracker Pro — Cloud Edition")
